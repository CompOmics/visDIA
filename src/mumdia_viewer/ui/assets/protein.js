/* mumdia-viewer: the protein page (ui/protein.py), linked panels in the manner of PeptideShaker.
 *
 * Panels: "pep" (every peptide of the group, grid pp-pep-grid), "pre" (the precursors of
 * the selected peptide, pp-pre-grid) and, on the search page, "find" (pp-find-grid).
 * A pick of a peptide (a click, the arrow keys, or a click on the sequence coverage)
 * selects it everywhere: the peptides grid, the coverage outline (window.mvCoverage),
 * the row of the peptides-by-runs heatmap, the address, and the precursors panel, whose
 * rows the server sends (store pp-pre-req, answered in pp-pre-data) and this file
 * keeps. A precursor row, or a peptide row's own precursor, opens its precursor page
 * on a double click, Enter or the open icon. Every number shown is the server's; this
 * file only keeps the panels linked.
 */

(function () {
  "use strict";

  var GRIDS = { pep: "pp-pep-grid", pre: "pp-pre-grid", find: "pp-find-grid" };
  var NOUN = { pep: "peptides", pre: "precursor rows" };
  var DEBOUNCE_MS = 150;
  var S = {
    page: null,
    sel: null,
    member: null,
    n: 0,
    pending: null,
    cache: new Map(),
    timers: {},
    fitted: {},
    posKey: null,
  };

  function NO() {
    return window.dash_clientside.no_update;
  }

  function h() {
    return window.React.createElement.apply(window.React, arguments);
  }

  // Whether Dash's layout holds a component with this id (a store has no element).
  function known(id) {
    try {
      var st = window.store && window.store.getState();
      return !st || !st.paths || !st.paths.strs || !!st.paths.strs[id];
    } catch (e) {
      return true;
    }
  }

  function setProps(id, props) {
    if (window.dash_clientside && window.dash_clientside.set_props && known(id)) {
      window.dash_clientside.set_props(id, props);
    }
  }

  function root() {
    return document.getElementById("pp-root");
  }

  // A new page (the server's token): forget the old page's selection and answers.
  function sync() {
    var r = root();
    var page = r ? r.getAttribute("data-page") : null;
    if (page !== S.page) {
      S.page = page;
      S.cache = new Map();
      S.pending = null;
      S.fitted = {};
      S.posKey = null;
      S.sel = r && r.getAttribute("data-peptide") ? r.getAttribute("data-peptide") : null;
      S.member = r ? r.getAttribute("data-member") || null : null;
      if (page) {
        setTimeout(boot, 0);
      }
    }
    return r;
  }

  // A new page: its grids come with their rows, so they are set up here (a grid's
  // eventListeners are attached at grid ready, after its first rows were drawn).
  function withApi(id, fn, tries) {
    var r = root();
    if (!r || (!document.getElementById(id) && !r.querySelector("[data-grid='" + id + "']"))) {
      // The grid component loads lazily: wait for it (about 4 s at most).
      if ((tries || 0) < 80 && r) {
        setTimeout(function () {
          withApi(id, fn, (tries || 0) + 1);
        }, 50);
      }
      return;
    }
    if (!window.dash_ag_grid || !window.dash_ag_grid.getApiAsync) {
      return;
    }
    window.dash_ag_grid.getApiAsync(id).then(function (api) {
      if (api && !(api.isDestroyed && api.isDestroyed())) {
        fn(api);
      }
    });
  }

  function boot() {
    var page = S.page;
    withApi(GRIDS.pep, function (api) {
      if (page !== S.page) {
        return;
      }
      fit("pep", api);
      if (S.sel !== null) {
        var node = api.getRowNode(S.sel);
        if (node) {
          node.setSelected(true, true);
          api.ensureNodeVisible(node, "middle");
        }
        // The page comes without the precursors of its selected peptide: ask for them.
        selectPep(S.sel, { force: true, now: true });
      }
      refreshPositions();
    });
    withApi(GRIDS.pre, function (api) {
      if (page !== S.page) {
        return;
      }
      fit("pre", api);
      if (!api.getSelectedNodes().length) {
        var first = api.getDisplayedRowAtIndex(0);
        if (first) {
          first.setSelected(true, true);
        }
      }
    });
    withApi(GRIDS.find, function (api) {
      fit("find", api);
    });
    if (S.sel !== null) {
      setTimeout(function () {
        highlightMatrix(S.sel);
        scrollCoverage();
      }, 400);
    }
  }

  function gridApi(id) {
    if (!document.getElementById(id) || !window.dash_ag_grid) {
      return null;
    }
    try {
      return window.dash_ag_grid.getApi(id);
    } catch (e) {
      return null;
    }
  }

  function go(url) {
    if (!url) {
      return;
    }
    window.history.pushState({}, "", url);
    window.dispatchEvent(new CustomEvent("_dashprivate_pushstate"));
    window.scrollTo(0, 0);
  }

  function stopLabel(t) {
    var x = Number(t);
    return x < 0.001 ? x.toExponential(0) : String(x);
  }

  function fmtInt(n) {
    return Number(n).toLocaleString("en-US");
  }

  function currentT() {
    var api = gridApi(GRIDS.pep) || gridApi(GRIDS.pre) || gridApi(GRIDS.find);
    var ctx = api ? api.getGridOption("context") : null;
    return ctx && ctx.threshold !== undefined ? ctx.threshold : null;
  }

  // ------------------------------------------------------------------ counts

  function passCount(rows, column, t) {
    var n = 0;
    (rows || []).forEach(function (r) {
      var v = r[column];
      if (r.label !== "decoy" && v !== null && v !== undefined && Number(v) <= Number(t)) {
        n += 1;
      }
    });
    return n;
  }

  function rowsOf(api) {
    var rows = [];
    if (api) {
      api.forEachNode(function (node) {
        if (node.data) {
          rows.push(node.data);
        }
      });
    }
    return rows;
  }

  function updateCount(panel, t) {
    var api = gridApi(GRIDS[panel]);
    var r = root();
    if (!api || !r || t === null || t === undefined) {
      return;
    }
    var column = panel === "pep" ? "peptide_q_value" : r.getAttribute("data-pre-mark") || "q_value";
    var rows = rowsOf(api);
    var n = passCount(rows, column, t);
    setProps("pp-" + panel + "-count", { children: fmtInt(n) + " of " + fmtInt(rows.length) });
    setProps("pp-" + panel + "-countbox", {
      title:
        fmtInt(n) + " of " + fmtInt(rows.length) + " " + NOUN[panel] + " pass " + column + " ≤ " +
        stopLabel(t) + " (the validation marks' test). The table lists every row of its parent, at any q.",
    });
  }

  // ------------------------------------------------------------------ selection

  function loading(on) {
    var el = document.getElementById("pp-panel-pre");
    if (el) {
      el.classList.toggle("pp-loading", !!on);
    }
  }

  function writeAddress() {
    var r = root();
    if (!r) {
      return;
    }
    var q = new URLSearchParams();
    var group = r.getAttribute("data-group");
    if (group) {
      q.set("group", group);
    }
    if (S.sel !== null && S.sel !== undefined && S.sel !== "") {
      q.set("peptide", S.sel);
    }
    if (S.member && group && group.indexOf(";") >= 0) {
      q.set("member", S.member);
    }
    var url = window.location.pathname + "?" + q.toString();
    if (url !== window.location.pathname + window.location.search) {
      window.history.replaceState(window.history.state, "", url);
    }
  }

  function accent() {
    var dark = document.documentElement.getAttribute("data-mantine-color-scheme") === "dark";
    return dark ? "#91a7ff" : "#4263eb";
  }

  // The selected peptide's row of the heatmap: an outline, scrolled into view.
  function highlightMatrix(key) {
    var box = document.getElementById("pp-matrix-box");
    var gd = box ? box.querySelector(".js-plotly-plot") : null;
    if (!gd || !gd.layout || !window.Plotly) {
      return;
    }
    var rows = (gd.layout.meta && gd.layout.meta.rows) || [];
    var i = rows.indexOf(Number(key));
    var shapes = (gd.layout.shapes || []).filter(function (s) {
      return s.name !== "selected";
    });
    if (i >= 0) {
      shapes.push({
        type: "rect",
        xref: "paper",
        yref: "y",
        x0: 0,
        x1: 1,
        y0: i - 0.5,
        y1: i + 0.5,
        line: { color: accent(), width: 2 },
        fillcolor: "rgba(0,0,0,0)",
        layer: "above",
        name: "selected",
      });
    }
    try {
      window.Plotly.relayout(gd, { shapes: shapes });
    } catch (e) {
      return;
    }
    if (i >= 0 && box.scrollHeight > box.clientHeight) {
      var n = rows.length || 1;
      var plot = gd.querySelector(".nsewdrag") || gd;
      var top = plot.getBoundingClientRect().top - gd.getBoundingClientRect().top;
      var h0 = plot.getBoundingClientRect().height;
      var y = top + ((i + 0.5) / n) * h0;
      if (y < box.scrollTop + 20 || y > box.scrollTop + box.clientHeight - 20) {
        box.scrollTop = Math.max(0, y - box.clientHeight / 2);
      }
    }
  }

  // The coverage's sequence text scrolls to the selected peptide (inside its own box).
  function scrollCoverage() {
    setTimeout(function () {
      var seq = document.querySelector("#pp-cov-body .mvc-seq");
      var el = seq ? seq.querySelector(".mvc-res.is-sel") : null;
      if (!seq || !el) {
        return;
      }
      var top = el.offsetTop - seq.offsetTop;
      if (top < seq.scrollTop || top > seq.scrollTop + seq.clientHeight - 24) {
        seq.scrollTop = Math.max(0, top - seq.clientHeight / 3);
      }
    }, 60);
  }

  function subjectText(row) {
    if (!row) {
      return "";
    }
    if (row.sequence) {
      return String(row.sequence);
    }
    var text = String(row.peptidoform || "");
    if (window.mvMods && window.mvMods.parse) {
      var pf = window.mvMods.parse(text);
      return (pf.decoy ? "DECOY_" : "") + pf.residues.map(function (r) {
        return r.aa;
      }).join("");
    }
    return text;
  }

  function applyPart(part) {
    if (!part) {
      return;
    }
    setProps(GRIDS.pre, { rowData: part.rows || [] });
    setProps("pp-pre-count", { children: part.count || "0" });
    setProps("pp-pre-countbox", { title: part.tip || "" });
    if (part.help) {
      setProps("pp-pre-help", { label: part.help });
    }
    loading(false);
  }

  function selectPep(key, opts) {
    sync();
    opts = opts || {};
    if (key === null || key === undefined || key === "") {
      return;
    }
    key = String(key);
    var api = gridApi(GRIDS.pep);
    var node = api ? api.getRowNode(key) : null;
    if (node) {
      if (!node.isSelected()) {
        node.setSelected(true, true);
      }
      if (opts.reveal) {
        api.ensureNodeVisible(node, "middle");
      }
    }
    if (key === S.sel && !opts.force) {
      return;
    }
    S.sel = key;
    if (node && node.data) {
      setProps("pp-pre-subject", { children: subjectText(node.data) });
    }
    if (window.mvCoverage) {
      window.mvCoverage.select(key);
    }
    scrollCoverage();
    highlightMatrix(key);
    writeAddress();
    setProps("pp-sel", { data: { peptide: Number(key), member: S.member } });
    clearTimeout(S.timers.pre);
    var kept = S.cache.get(key);
    if (kept) {
      applyPart(kept);
      return;
    }
    loading(true);
    S.timers.pre = setTimeout(
      function () {
        S.n += 1;
        S.pending = key;
        setProps("pp-pre-req", { data: { peptide: Number(key), n: S.n, t: currentT() } });
      },
      opts.now ? 0 : DEBOUNCE_MS
    );
  }

  function open(url) {
    if (url) {
      go(url);
    }
  }

  function selectedRow(api) {
    var nodes = api && api.getSelectedNodes ? api.getSelectedNodes() : [];
    return nodes && nodes.length && nodes[0].data ? nodes[0].data : null;
  }

  // ------------------------------------------------------------------ narrow grids

  // A grid whose columns do not fit hides the columns that allow it (context.hideOrder,
  // the highest first), once per grid and page.
  function fit(panel, api, tries) {
    if (!api || S.fitted[panel]) {
      return;
    }
    var el = document.getElementById(GRIDS[panel]);
    var viewport = el && el.querySelector(".ag-center-cols-viewport");
    var avail = viewport ? viewport.clientWidth : 0;
    if (!avail) {
      // Not laid out yet: try again shortly.
      if ((tries || 0) < 20) {
        setTimeout(function () {
          fit(panel, api, (tries || 0) + 1);
        }, 50);
      }
      return;
    }
    S.fitted[panel] = true;
    var cols = api.getAllDisplayedColumns();
    var total = 0;
    cols.forEach(function (c) {
      if (!c.getPinned()) {
        var d = c.getColDef() || {};
        total += d.flex ? Math.max(d.minWidth || 0, (d.context && d.context.fitWidth) || 0) : c.getActualWidth();
      }
    });
    var cand = cols
      .filter(function (c) {
        var d = c.getColDef() || {};
        return d.context && d.context.hideOrder;
      })
      .sort(function (a, b) {
        return b.getColDef().context.hideOrder - a.getColDef().context.hideOrder;
      });
    var hide = [];
    while (total > avail + 1 && cand.length) {
      var c = cand.shift();
      hide.push(c.getColId());
      total -= c.getActualWidth();
    }
    if (hide.length) {
      api.setColumnsVisible(hide, false);
    }
  }

  // ------------------------------------------------------------------ positions

  // The positions of the peptides follow the member the coverage shows (its spans).
  function refreshPositions() {
    var bar = document.querySelector("#pp-cov-body .mvc-bar");
    var api = gridApi(GRIDS.pep);
    if (!bar || !api || !api.getColumn("_pos")) {
      return;
    }
    var member = bar.getAttribute("data-member") || "";
    var key = S.page + "|" + member + "|" + bar.getAttribute("data-spans").length;
    if (key === S.posKey) {
      return;
    }
    S.posKey = key;
    var spans = [];
    try {
      spans = JSON.parse(bar.getAttribute("data-spans") || "[]");
    } catch (e) {
      spans = [];
    }
    var pos = {};
    spans.forEach(function (s) {
      var id = String(s[2]);
      var hit = pos[id];
      if (!hit) {
        pos[id] = [s[0] + 1, s[1], 1];
      } else {
        hit[2] += 1;
        if (s[0] + 1 < hit[0]) {
          hit[0] = s[0] + 1;
          hit[1] = s[1];
        }
      }
    });
    var update = [];
    api.forEachNode(function (node) {
      if (!node.data) {
        return;
      }
      var hit = pos[String(node.data.base_peptide_id)];
      var start = hit ? hit[0] : null;
      if (node.data._start !== start || node.data._n_pos !== (hit ? hit[2] : 0)) {
        update.push(
          Object.assign({}, node.data, {
            _start: start,
            _end: hit ? hit[1] : null,
            _n_pos: hit ? hit[2] : 0,
          })
        );
      }
    });
    if (update.length) {
      api.applyTransaction({ update: update });
    }
    if (member && member !== S.member) {
      S.member = member;
      writeAddress();
    }
  }

  // A click on a heatmap cell opens the precursor page of its precursor (customdata[1]).
  function bindMatrix() {
    var gd = document.querySelector("#pp-matrix-box .js-plotly-plot");
    if (!gd || gd.__ppClick || typeof gd.on !== "function") {
      return;
    }
    gd.__ppClick = true;
    gd.on("plotly_click", function (ev) {
      var p = ev && ev.points && ev.points[0];
      var data = p && p.customdata;
      var url = data && data.length > 1 ? data[1] : null;
      if (url) {
        go(url);
      }
    });
  }

  var observer = new MutationObserver(function () {
    sync();
    bindMatrix();
    clearTimeout(S.timers.pos);
    S.timers.pos = setTimeout(refreshPositions, 30);
  });
  function startObserver() {
    observer.observe(document.body, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ["data-spans", "data-member"],
    });
  }
  if (document.body) {
    startObserver();
  } else {
    document.addEventListener("DOMContentLoaded", startObserver);
  }

  // ------------------------------------------------------------------ coverage and members

  document.addEventListener("mv-coverage-pick", function (ev) {
    var d = ev.detail || {};
    var host = d.el && d.el.closest ? d.el.closest("#pp-cov-body") : null;
    if (!host || !sync()) {
      return;
    }
    selectPep(d.pep, { reveal: true, now: true });
  });

  document.addEventListener("mv-coverage-member", function (ev) {
    var d = ev.detail || {};
    var host = d.el && d.el.closest ? d.el.closest("#pp-cov-body") : null;
    if (host && d.member) {
      S.member = d.member;
      writeAddress();
    }
  });

  // A member of the members card shows its coverage.
  document.addEventListener("click", function (ev) {
    var btn = ev.target && ev.target.closest ? ev.target.closest(".pp-mbtn") : null;
    if (!btn || btn.disabled || !sync()) {
      return;
    }
    var member = btn.getAttribute("data-member");
    S.member = member;
    writeAddress();
    S.n += 1;
    setProps("pp-cov-req", {
      data: {
        group: btn.getAttribute("data-group"),
        member: member,
        peptide: S.sel !== null ? Number(S.sel) : null,
        t: currentT(),
        n: S.n,
      },
    });
    var card = document.getElementById("pp-cov-card");
    if (card && card.getBoundingClientRect().top < 0) {
      card.scrollIntoView({ block: "start", behavior: "smooth" });
    }
  });

  // ------------------------------------------------------------------ renderers

  var R = {
    // Position of the peptide in the shown member (viewer-derived, from the coverage).
    PpPos: function (p) {
      if (!p.data) {
        return null;
      }
      if (p.data._start === null || p.data._start === undefined) {
        return h(
          "span",
          { className: "ib-dim pp-pos-none", title: "Not found in the sequence of the shown member" },
          "not found"
        );
      }
      var kids = [h("span", { key: "a", className: "pp-pos" }, p.data._start + "–" + p.data._end)];
      if (p.data._n_pos > 1) {
        kids.push(
          h(
            "span",
            { key: "n", className: "pp-pos-more", title: "Found " + p.data._n_pos + " times in the sequence" },
            "+" + (p.data._n_pos - 1)
          )
        );
      }
      return h("span", { className: "pp-pos-cell" }, kids);
    },

    // In the rollup (top-N sum) of the protein quantity: its rank.
    PpRollup: function (p) {
      if (!p.data || p.data._rollup_rank === null || p.data._rollup_rank === undefined) {
        return null;
      }
      return h(
        "span",
        {
          className: "pp-roll",
          title:
            "In the sum that gives the protein quantity (rank " + p.data._rollup_rank +
            " of the per-peptide maxima; viewer-derived)",
        },
        "#" + p.data._rollup_rank
      );
    },
  };
  window.dashAgGridComponentFunctions = Object.assign(window.dashAgGridComponentFunctions || {}, R);

  // ------------------------------------------------------------------ grid events

  var F = {
    ppRowClicked: function (params, panel) {
      var row = params && params.data;
      if (!row) {
        return;
      }
      var target = params.event && params.event.target;
      if (target && target.closest && target.closest(".ib-open")) {
        return;
      }
      if (panel === "pep") {
        clearTimeout(S.timers.focus);
        selectPep(row._key, { now: true });
      } else if (panel === "find") {
        open(row._protein);
      }
    },

    ppRowDoubleClicked: function (params, panel) {
      var row = params && params.data;
      if (row) {
        open(panel === "find" ? row._protein : row._href);
      }
    },

    ppKeyDown: function (params, panel) {
      var ev = params && params.event;
      if (!ev) {
        return;
      }
      if (ev.key === "Enter") {
        ev.preventDefault();
        var row = selectedRow(params.api) || params.data;
        if (row) {
          open(panel === "find" ? row._protein : row._href);
        }
      } else if (ev.key === " ") {
        ev.preventDefault();
      }
    },

    // The arrow keys move the selection with the focus (the precursors wait for a pause).
    ppFocused: function (params, panel) {
      if (!params || params.rowIndex === null || params.rowIndex === undefined || params.rowIndex < 0) {
        return;
      }
      var el = document.getElementById(GRIDS[panel]);
      if (!el || !el.contains(document.activeElement) || params.rowPinned) {
        return;
      }
      var node = params.api.getDisplayedRowAtIndex(params.rowIndex);
      if (!node || !node.data) {
        return;
      }
      if (!node.isSelected()) {
        node.setSelected(true, true);
      }
      if (panel === "pep") {
        clearTimeout(S.timers.focus);
        S.timers.focus = setTimeout(function () {
          selectPep(node.data._key);
        }, DEBOUNCE_MS);
      }
    },

    ppFirstData: function (params, panel) {
      sync();
      var api = params && params.api;
      if (!api) {
        return;
      }
      fit(panel, api);
      if (panel === "pep" && S.sel !== null) {
        var node = api.getRowNode(S.sel);
        if (node) {
          node.setSelected(true, true);
          api.ensureNodeVisible(node, "middle");
        }
        refreshPositions();
      }
      if (panel === "pre" && !api.getSelectedNodes().length) {
        var first = api.getDisplayedRowAtIndex(0);
        if (first) {
          first.setSelected(true, true);
        }
      }
    },

    ppRowDataUpdated: function (params, panel) {
      var api = params && params.api;
      if (!api || panel !== "pre") {
        return;
      }
      setTimeout(function () {
        if (api.isDestroyed && api.isDestroyed()) {
          return;
        }
        var first = api.getDisplayedRowAtIndex(0);
        if (first && !api.getSelectedNodes().length) {
          first.setSelected(true, true);
        }
        api.ensureIndexVisible(0, "top");
      }, 0);
    },
  };
  window.dashAgGridFunctions = Object.assign(window.dashAgGridFunctions || {}, F);

  // ------------------------------------------------------------------ callbacks

  window.dash_clientside = Object.assign({}, window.dash_clientside, {
    mvp: {
      // The server's precursors of a peptide: shown when it is still the selection.
      applyPre: function (data) {
        sync();
        if (!data || data.peptide === null || data.peptide === undefined) {
          return NO();
        }
        var key = String(data.peptide);
        S.cache.set(key, data);
        if (key === S.sel) {
          applyPart(data);
        }
        return NO();
      },

      // The header threshold: every mark, bar colour and count follows.
      threshold: function (t) {
        sync();
        if (t === null || t === undefined) {
          return NO();
        }
        Object.keys(GRIDS).forEach(function (p) {
          var api = gridApi(GRIDS[p]);
          if (!api) {
            return;
          }
          var ctx = api.getGridOption("context");
          if (ctx) {
            ctx.threshold = t;
          } else {
            api.setGridOption("context", { threshold: t });
          }
          var col = api.getColumn("_valid");
          var def = col && col.getColDef();
          var tpl = def && def.cellRendererParams && def.cellRendererParams.tipTemplate;
          if (tpl) {
            def.headerTooltip = tpl.replace("{t}", stopLabel(t));
            api.refreshHeader();
          }
          api.refreshCells({ force: true });
        });
        updateCount("pep", t);
        updateCount("pre", t);
        // A cached precursor answer counts at its own threshold: ask again.
        S.cache = new Map();
        return NO();
      },

      // The search text lives in the address.
      searchAddress: function (value) {
        var r = root();
        if (!r || !r.getAttribute("data-search")) {
          return NO();
        }
        var q = new URLSearchParams();
        var text = String(value || "").trim();
        if (text) {
          q.set("search", text);
        }
        var url = window.location.pathname + (text ? "?" + q.toString() : "");
        if (url !== window.location.pathname + window.location.search) {
          window.history.replaceState(window.history.state, "", url);
        }
        return NO();
      },
    },
  });

  window.mvpInternals = { passCount: passCount, stopLabel: stopLabel, state: S };
})();
