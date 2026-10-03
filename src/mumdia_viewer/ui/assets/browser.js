/* mumdia-viewer: the identification page (linked panels in the manner of PeptideShaker).
 *
 * AG Grid cell renderers (window.dashAgGridComponentFunctions), grid functions and event
 * handlers (window.dashAgGridFunctions), the page's client-side callbacks
 * (dash_clientside.mvb), the selection, the child panels, the preview and the address.
 * Every number shown is the server's; the code here only formats it, keeps the panels
 * linked and decides when to ask the server.
 *
 * Panels: "top" (the first table, grid ib-grid), "pep" (the peptides of the selected
 * protein group, ib-pep-grid) and "pre" (the precursors of the selected peptide,
 * ib-pre-grid). A pick in a panel selects its row; a parent asks for its children
 * (store ib-need, answered in ib-children) unless the answer is kept here already, and
 * the preview follows the selected precursor (ib-prec, answered in ib-preview-data).
 *
 * Responsiveness. Each Dash component of a newly inserted card dispatches one store
 * action, and every action runs the selectors of every mounted component, so a card of
 * about 300 components blocked the main thread for about one second. The preview is
 * therefore asked for when the selection rests, inserted after the child panels have
 * painted and while no key is held, and its large static tables (the ion ladder, the
 * sequence diagram) are drawn as plain HTML ("islands", see islandize): a Mantine
 * tooltip inside them becomes the element's title.
 */

(function () {
  "use strict";

  var GRIDS = { top: "ib-grid", pep: "ib-pep-grid", pre: "ib-pre-grid" };
  var PANELS = { top: "ib-panel-top", pep: "ib-panel-pep", pre: "ib-panel-pre" };
  var MENUS = { top: "ib-cols", pep: "ib-pep-cols", pre: "ib-pre-cols" };
  var CONTROLS = ["q", "search", "charge", "protein", "mod", "quant", "run", "decoys"];
  var CONTROL_DEFAULTS = {
    q: null, // the unit's default q column (defaultQ)
    search: "",
    charge: "any",
    protein: "",
    mod: "",
    quant: "any",
    run: "all",
    decoys: false,
  };
  var UNIT_Q = { precursor: "precursor_q", peptide: "peptide_q_value", protein_group: "pg_q_value" };
  var NOUN = { pep: "peptides", pre: "precursor rows" };
  var QUANT_WORDS = {
    quantified: "quantified",
    not_quantifiable: "not quantifiable",
    not_selected: "not selected",
  };
  var SPECIES = {
    HUMAN: "indigo",
    YEAST: "yellow",
    ECOLI: "teal",
    MOUSE: "pink",
    RAT: "grape",
    BOVIN: "orange",
    ARATH: "lime",
    DROME: "cyan",
    CAEEL: "violet",
  };
  var FIXED = { _valid: true, _open: true };
  var NAV_KEYS = {
    ArrowUp: true,
    ArrowDown: true,
    PageUp: true,
    PageDown: true,
    Home: true,
    End: true,
  };
  // The child queries wait for a pause of the keys; the preview waits longer.
  var DEBOUNCE_MS = 150;
  var PREVIEW_KEY_DELAY = 350;
  var KEY_QUIET = 320;
  var PREFETCH_DELAY = 650;
  var ROW_HEIGHT = 26;
  var COLLAPSE_KEY = "mumdia-viewer.ib.preview-collapsed";

  // ------------------------------------------------------------------ small LRU map

  function LRU(max) {
    this.max = max;
    this.map = new Map();
  }
  LRU.prototype.get = function (key) {
    if (!this.map.has(key)) {
      return undefined;
    }
    var v = this.map.get(key);
    this.map.delete(key);
    this.map.set(key, v);
    return v;
  };
  LRU.prototype.has = function (key) {
    return this.map.has(key);
  };
  LRU.prototype.set = function (key, value) {
    this.map.delete(key);
    this.map.set(key, value);
    while (this.map.size > this.max) {
      this.map.delete(this.map.keys().next().value);
    }
  };

  // Kept for the session (the result set does not change): child panels and previews.
  var C = new LRU(80);

  // Page state. `view` is the first table's filters, `sel` the selection, `written` the
  // address query this page wrote last, `built` the query the page was built from.
  var S = {
    page: null,
    synced: null,
    view: null,
    sel: {},
    built: null,
    written: null,
    pending: {},
    timers: {},
    loadTimers: {},
    waiting: {},
    parts: {},
    nonce: 0,
    need: null,
    appliedN: null,
    decoysChanged: false,
    tChanged: false,
    started: false,
    startTimer: null,
    prec0: null,
    addrSel: false,
    lastKey: 0,
    lastPanel: "top",
    dir: 1,
    focusPending: {},
    pfTimer: null,
    needTimer: null,
    pfN: 0,
    locN: 0,
    sortPending: false,
    narrowDone: {},
    locShown: false,
  };

  // The preview: the wanted precursor, the cards kept, the card shown.
  var PV = {
    want: null,
    n: 0,
    cards: new LRU(16),
    timer: null,
    insTimer: null,
    shown: null,
    collapsed: false,
    seq: 0,
  };
  try {
    PV.collapsed = window.localStorage.getItem(COLLAPSE_KEY) === "1";
  } catch (e) {
    PV.collapsed = false;
  }

  function NO() {
    return window.dash_clientside.no_update;
  }

  function h() {
    return window.React.createElement.apply(null, arguments);
  }

  function shared() {
    return window.dashAgGridComponentFunctions || {};
  }

  function gridApi(id) {
    try {
      var api = window.dash_ag_grid && window.dash_ag_grid.getApi(id);
      return api && !api.isDestroyed() ? api : null;
    } catch (e) {
      return null;
    }
  }

  function root() {
    return document.getElementById("ib-root");
  }

  // Whether Dash's layout has a component with this id (a store renders no element).
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

  // The q column the data layer uses when none is chosen (browser.default_q).
  function defaultQ(view) {
    if (view.unit === "precursor" && view.run) {
      return "run_psm_q";
    }
    return UNIT_Q[view.unit] || "precursor_q";
  }

  function isNum(v) {
    return v !== null && v !== undefined && v !== "";
  }

  function same(a, b) {
    return isNum(a) && isNum(b) && String(a) === String(b);
  }

  function now() {
    return Date.now();
  }

  function whichOf(panel) {
    if (panel === "pep") {
      return "peptide";
    }
    if (panel === "pre") {
      return "precursor";
    }
    return S.view ? S.view.unit : null;
  }

  // ------------------------------------------------------------------ navigation

  function onThisPage() {
    return /identifications\/?$/.test(window.location.pathname);
  }

  // Leaving the page: the picks and requests that wait for a pause go with it.
  function cancelPending() {
    Object.keys(S.timers).forEach(function (k) {
      clearTimeout(S.timers[k]);
    });
    S.timers = {};
    S.focusPending = {};
    clearTimeout(S.pfTimer);
    clearTimeout(S.needTimer);
    clearTimeout(PV.timer);
    clearTimeout(PV.insTimer);
  }

  // In-app navigation, as dcc.Link does it: the router builds the page, no reload.
  function go(url) {
    if (!url) {
      return;
    }
    cancelPending();
    window.history.pushState({}, "", url);
    window.dispatchEvent(new CustomEvent("_dashprivate_pushstate"));
    window.scrollTo(0, 0);
  }

  function currentSort() {
    var api = gridApi(GRIDS.top);
    if (api) {
      var state = api.getColumnState().filter(function (c) {
        return c.sort;
      });
      if (state.length) {
        return { col: state[0].colId, desc: state[0].sort === "desc" };
      }
    }
    var v = S.view || {};
    return { col: v.sort || "score", desc: v.desc !== false };
  }

  // The address query of a view and a selection; the same keys and defaults as
  // browser.Filters.query and browser.Selection.query.
  function addressOf(view, sel, sort) {
    var p = new URLSearchParams();
    if (view.unit && view.unit !== "protein_group") {
      p.set("unit", view.unit);
    }
    ["q", "search", "charge", "protein", "mod", "quant"].forEach(function (k) {
      if (view[k]) {
        p.set(k, view[k]);
      }
    });
    if (view.run) {
      p.set("in_run", view.run);
    }
    if (view.decoys) {
      p.set("decoys", "1");
    }
    if (sort && (sort.col !== "score" || !sort.desc)) {
      p.set("sort", sort.col);
      p.set("order", sort.desc ? "desc" : "asc");
    }
    p.set("t", String(view.t));
    if (sel) {
      if (view.unit === "protein_group" && sel.group) {
        p.set("group", sel.group);
      }
      if (view.unit !== "precursor" && isNum(sel.peptide)) {
        p.set("peptide", String(sel.peptide));
      }
      if (isNum(sel.cid)) {
        if (sel.run) {
          p.set("run", sel.run);
        }
        p.set("cid", String(sel.cid));
      }
    }
    return "?" + p.toString();
  }

  // An address in a comparable form: the first value of each key, empty values left
  // out (as the router's query_of reads it), sorted by key.
  function norm(search) {
    var p = new URLSearchParams(search || "");
    var seen = {};
    var out = [];
    p.forEach(function (v, k) {
      if (seen[k] || v === "") {
        return;
      }
      seen[k] = true;
      out.push(k + "=" + v);
    });
    return out.sort().join("&");
  }

  function builtNorm() {
    var q = S.built || {};
    return Object.keys(q)
      .filter(function (k) {
        return q[k] !== "" && q[k] !== null && q[k] !== undefined;
      })
      .map(function (k) {
        return k + "=" + q[k];
      })
      .sort()
      .join("&");
  }

  function writeAddress() {
    if (!root() || !S.view || !onThisPage()) {
      return; // (a navigation to another page may have started)
    }
    var search = addressOf(S.view, S.sel, currentSort());
    S.written = search;
    if (window.location.search !== search) {
      window.history.replaceState(
        window.history.state,
        "",
        window.location.pathname + search + window.location.hash
      );
    }
  }

  // A link to the address the router built last does not change the router's input, so
  // the router does not rebuild the page: this page asks the server to build it.
  function onAddress() {
    setTimeout(function () {
      if (!root() || !S.view) {
        return;
      }
      if (!onThisPage()) {
        return;
      }
      var now_ = norm(window.location.search);
      if (S.written !== null && now_ === norm(S.written)) {
        return;
      }
      if (now_ !== builtNorm()) {
        return; // the router sees a new address and builds the page
      }
      S.written = window.location.search;
      setProps("ib-address", { data: window.location.search });
    }, 0);
  }
  window.addEventListener("_dashprivate_pushstate", onAddress);
  window.addEventListener("popstate", onAddress);

  // The header search, while this page is shown. This page rewrites the address with
  // history.replaceState, which dcc.Location does not see; with refresh=True the shell's
  // Location then reloads its stale address instead of the search. So this page sends
  // the search itself, as an in-app navigation (see the shared change proposed for
  // app.py: refresh="callback-nav").
  document.addEventListener(
    "keydown",
    function (event) {
      var target = event.target;
      if (event.key !== "Enter" || !target || target.id !== "global-search") {
        return;
      }
      var el = root();
      var text = String(target.value || "").trim();
      if (!el || !text) {
        return;
      }
      event.preventDefault();
      event.stopPropagation();
      target.blur();
      go((el.dataset.base || "/") + "identifications?search=" + encodeURIComponent(text));
    },
    true
  );

  // Clicks on the page's plain buttons (no Dash callback): the threshold pill opens the
  // header control, the preview handle, the notice's close, "show the selected row",
  // and the "more" of a clamped note.
  document.addEventListener("click", function (event) {
    var t = event.target;
    if (!t || !t.closest) {
      return;
    }
    if (t.closest(".ib-t")) {
      var select = document.getElementById("q-select");
      if (select) {
        select.focus();
        select.click();
      }
      return;
    }
    if (t.closest("#ib-pv-toggle")) {
      toggleCollapsed();
      return;
    }
    if (t.closest("#ib-notice-close")) {
      hideNotice();
      return;
    }
    if (t.closest("#ib-locate")) {
      showSelected();
      return;
    }
    var more = t.closest(".ib-clamp-more");
    if (more) {
      var box = more.closest(".ib-clamp");
      if (box) {
        box.classList.toggle("ib-clamp-open");
        more.textContent = box.classList.contains("ib-clamp-open") ? "less" : "more";
      }
    }
  });

  // ------------------------------------------------------------------ formatting

  function fmtInt(v) {
    return Number(v).toLocaleString("en-US");
  }

  function fmtNumber(v, kind) {
    if (v === null || v === undefined || v === "") {
      return "";
    }
    var n = Number(v);
    if (!isFinite(n)) {
      return String(v);
    }
    if (kind === "rt") {
      return n.toFixed(1);
    }
    if (kind === "int") {
      return fmtInt(n);
    }
    if (kind === "id") {
      return String(v);
    }
    if (kind === "q") {
      return window.mvFormat ? window.mvFormat(n, "q") : String(n);
    }
    if (n !== 0 && (Math.abs(n) < 0.001 || Math.abs(n) >= 1e6)) {
      return n.toExponential(3);
    }
    return Number(n.toPrecision(4)).toString();
  }

  // The header's threshold label (state.stop_label): 1e-4, 5e-4, 0.001, ... 0.1.
  function stopLabel(t) {
    var x = Number(t);
    return x < 0.001 ? x.toExponential(0) : String(x);
  }

  // ------------------------------------------------------------------ renderers

  function skeleton(width) {
    return h("span", { className: "ib-skel", style: width ? { width: width } : undefined });
  }

  function member(name, key, first) {
    var text = String(name);
    var kids = [];
    if (text.indexOf("DECOY_") === 0) {
      kids.push(h("span", { key: "d", className: "mv-pep-decoy" }, "DECOY_"));
      text = text.slice(6);
    }
    kids.push(text);
    return h("span", { key: key, className: first ? "ib-member-first" : "ib-member" }, kids);
  }

  function speciesOf(text) {
    var out = [];
    String(text || "")
      .split(";")
      .forEach(function (m) {
        var name = m.replace(/^DECOY_/, "");
        var i = name.lastIndexOf("_");
        if (i <= 0) {
          return;
        }
        var sp = name.slice(i + 1);
        if (/^[A-Z][A-Z0-9]{1,5}$/.test(sp) && out.indexOf(sp) < 0) {
          out.push(sp);
        }
      });
    return out;
  }

  var R = {
    // The validation mark (the shared MvValidation); nothing while the row loads.
    IbValid: function (p) {
      if (!p.data || !shared().MvValidation) {
        return null;
      }
      return shared().MvValidation(p);
    },

    // An in-cell bar with its value (the shared MvBar). A missing quantity names its
    // quant state instead (never 0).
    IbBar: function (p) {
      if (!p.data) {
        return skeleton("60%");
      }
      if ((p.value === null || p.value === undefined) && p.colDef && p.colDef.field === "quantity") {
        var st = p.data.quant_state;
        return st && st !== "quantified"
          ? h("span", { className: "ib-noquant" }, QUANT_WORDS[st] || String(st))
          : null;
      }
      return shared().MvBar ? shared().MvBar(p) : fmtNumber(p.value);
    },

    IbPep: function (p) {
      if (!p.data) {
        return skeleton();
      }
      if (!p.value) {
        return null;
      }
      var el = window.mvPeptidoformElement
        ? window.mvPeptidoformElement(String(p.value))
        : String(p.value);
      return h("span", { className: "ib-pep" }, el);
    },

    // A protein group: the first member bold, the next ones as chips, then "+n".
    IbGroup: function (p) {
      if (!p.data) {
        return skeleton("70%");
      }
      var members = String(p.value || "")
        .split(";")
        .filter(Boolean);
      if (!members.length) {
        return h("span", { className: "ib-dim" }, "no protein group");
      }
      var kids = [member(members[0], "m0", true)];
      var budget = 30 - members[0].length;
      var shown = 1;
      for (var i = 1; i < members.length && i < 4; i++) {
        if (members[i].length + 3 > budget) {
          break;
        }
        budget -= members[i].length + 3;
        kids.push(member(members[i], "m" + i, false));
        shown += 1;
      }
      if (shown < members.length) {
        kids.push(
          h("span", { key: "more", className: "ib-member ib-member-more" }, "+" + (members.length - shown))
        );
      }
      return h("span", { className: "ib-group" }, kids);
    },

    // Species from the entry-name suffix (the viewer's rule, said in the header).
    IbSpecies: function (p) {
      if (!p.data) {
        return null;
      }
      var list = speciesOf(p.value);
      if (!list.length) {
        return null;
      }
      return h(
        "span",
        { className: "ib-species" },
        list.slice(0, 3).map(function (s) {
          return h(
            "span",
            {
              key: s,
              className: "ib-sp ib-sp-" + (SPECIES[s] || "gray"),
              title: "Species from the entry-name suffix _" + s + " (the viewer's rule)",
            },
            s
          );
        })
      );
    },

    IbOpen: function (p) {
      if (!p.data || !p.data._href) {
        return null;
      }
      var url = p.data._href;
      return h(
        "a",
        {
          className: "ib-open",
          href: url,
          title: "Open the precursor page",
          "aria-label": "Open the precursor page",
          tabIndex: -1,
          onClick: function (ev) {
            if (ev.button !== 0 || ev.metaKey || ev.ctrlKey || ev.shiftKey || ev.altKey) {
              return;
            }
            ev.preventDefault();
            ev.stopPropagation();
            go(url);
          },
        },
        h("span", { className: "ib-open-icon" })
      );
    },

    IbLabel: function (p) {
      if (!p.data) {
        return skeleton("44px");
      }
      var v = p.value || "";
      return h("span", { className: "ib-pill ib-pill-" + (v === "decoy" ? "decoy" : "target") }, v);
    },

    IbCharge: function (p) {
      if (!p.data || p.value === null || p.value === undefined) {
        return null;
      }
      return h("span", { className: "ib-charge" }, p.value + "+");
    },

    IbQuantState: function (p) {
      if (!p.data) {
        return null;
      }
      var v = p.value;
      if (!v) {
        return h("span", { className: "ib-dim" }, "no quant table");
      }
      return h(
        "span",
        { className: "ib-state" },
        h("span", { className: "ib-dot ib-dot-" + v }),
        h("span", null, QUANT_WORDS[v] || v)
      );
    },

    IbText: function (p) {
      if (!p.data) {
        return skeleton("64%");
      }
      return h("span", { className: "ib-text" }, p.value === null || p.value === undefined ? "" : String(p.value));
    },

    IbNumber: function (p) {
      if (!p.data) {
        return null;
      }
      return h("span", { className: "ib-num" }, fmtNumber(p.value, p.kind));
    },

    IbWinner: function (p) {
      if (!p.data || p.value === null || p.value === undefined) {
        return null;
      }
      return p.value
        ? h("span", { className: "ib-yes" }, "winner")
        : h(
            "span",
            {
              className: "ib-no",
              title: "The winning row of this group does not pass the table's filters (for example a decoy won)",
            },
            "not winner"
          );
    },

    IbFlag: function (p) {
      if (!p.data || !p.value) {
        return null;
      }
      return h("span", { className: "ib-pill ib-pill-" + (p.colour || "gray") }, p.text || "yes");
    },

    IbRun: function (p) {
      if (!p.data || !p.value) {
        return null;
      }
      return h("span", { className: "ib-run" }, String(p.value));
    },
  };
  window.dashAgGridComponentFunctions = Object.assign(window.dashAgGridComponentFunctions || {}, R);

  // ------------------------------------------------------------------ islands
  // A static subtree of the preview card (html components only, no id, no component
  // props; Mantine tooltips with a text label, except the info icons) is inserted as a
  // host element with the subtree's root props, and its children are built as plain
  // DOM. The page's look and the detail page's hover links (data-frag) stay; a tooltip
  // becomes the element's title.

  var HTML_NS = "dash_html_components";
  var ISLAND_MIN = 3;
  var ISLAND_TIPS_MIN = 12;
  var KEEP_TAGS = {
    A: 1,
    Button: 1,
    Input: 1,
    Select: 1,
    Option: 1,
    Textarea: 1,
    Form: 1,
    Iframe: 1,
    Video: 1,
    Audio: 1,
    Canvas: 1,
    Embed: 1,
    Object: 1,
    Script: 1,
    Style: 1,
    Link: 1,
    Meta: 1,
    Base: 1,
    Details: 1,
    Summary: 1,
    Dialog: 1,
  };
  var UNITLESS = {
    flex: 1,
    flexGrow: 1,
    flexShrink: 1,
    opacity: 1,
    order: 1,
    zIndex: 1,
    fontWeight: 1,
    lineHeight: 1,
    gridRow: 1,
    gridColumn: 1,
    gridRowStart: 1,
    gridRowEnd: 1,
    gridColumnStart: 1,
    gridColumnEnd: 1,
    zoom: 1,
    columnCount: 1,
  };
  var SKIP_PROPS = {
    children: 1,
    n_clicks: 1,
    n_clicks_timestamp: 1,
    disable_n_clicks: 1,
    key: 1,
    loading_state: 1,
    setProps: 1,
  };
  var memo = typeof WeakMap !== "undefined" ? new WeakMap() : null;

  function isComp(n) {
    return !!(n && typeof n === "object" && !Array.isArray(n) && n.type && n.namespace);
  }

  function plainProps(p) {
    for (var k in p) {
      if (!Object.prototype.hasOwnProperty.call(p, k) || k === "children") {
        continue;
      }
      var v = p[k];
      if (v === null || v === undefined) {
        continue;
      }
      if (k === "style") {
        if (typeof v !== "object" || Array.isArray(v)) {
          return false;
        }
        continue;
      }
      if (typeof v === "object" || typeof v === "function") {
        return false;
      }
    }
    return true;
  }

  function helpIcon(n) {
    return isComp(n) && /\bmv-help\b/.test(String((n.props || {}).className || ""));
  }

  // {n: components, tips: tooltips} of a static subtree, or null.
  function staticInfo(n) {
    if (n === null || n === undefined || typeof n !== "object") {
      return { n: 0, tips: 0 };
    }
    if (memo && memo.has(n)) {
      return memo.get(n);
    }
    var out = null;
    if (Array.isArray(n)) {
      out = { n: 0, tips: 0 };
      for (var i = 0; i < n.length; i++) {
        var c = staticInfo(n[i]);
        if (!c) {
          out = null;
          break;
        }
        out.n += c.n;
        out.tips += c.tips;
      }
    } else if (isComp(n)) {
      var p = n.props || {};
      if (p.id !== undefined && p.id !== null) {
        out = null;
      } else if (n.namespace === HTML_NS && !KEEP_TAGS[n.type] && plainProps(p)) {
        var k = staticInfo(p.children);
        out = k ? { n: 1 + k.n, tips: k.tips } : null;
      } else if (
        n.namespace === "dash_mantine_components" &&
        n.type === "Tooltip" &&
        typeof p.label === "string" &&
        !Array.isArray(p.children) &&
        isComp(p.children) &&
        !helpIcon(p.children)
      ) {
        var k2 = staticInfo(p.children);
        out = k2 ? { n: 1 + k2.n, tips: k2.tips + 1 } : null;
      }
    }
    if (memo && n && typeof n === "object") {
      memo.set(n, out);
    }
    return out;
  }

  function hasKids(c) {
    return c !== null && c !== undefined && !(Array.isArray(c) && c.length === 0) && c !== "";
  }

  // A copy of a component tree with its large static subtrees replaced by hosts; the
  // subtrees' children go into `out` by host key.
  function islandize(n, out) {
    if (Array.isArray(n)) {
      return n.map(function (c) {
        return islandize(c, out);
      });
    }
    if (!isComp(n)) {
      return n;
    }
    var p = n.props || {};
    if (n.namespace === HTML_NS && (p.id === undefined || p.id === null) && !KEEP_TAGS[n.type]) {
      var info = staticInfo(n);
      if (info && hasKids(p.children) && info.n >= (info.tips ? ISLAND_TIPS_MIN : ISLAND_MIN)) {
        PV.seq += 1;
        var key = "i" + PV.seq;
        out[key] = p.children;
        var hp = Object.assign({}, p);
        delete hp.children;
        hp["data-ib-island"] = key;
        return { namespace: n.namespace, type: n.type, props: hp };
      }
    }
    if (p.children === undefined || p.children === null) {
      return n;
    }
    var np = Object.assign({}, p, { children: islandize(p.children, out) });
    return Object.assign({}, n, { props: np });
  }

  function applyStyle(el, st) {
    Object.keys(st).forEach(function (k) {
      var v = st[k];
      if (v === null || v === undefined || v === "") {
        return;
      }
      if (k.indexOf("--") === 0) {
        el.style.setProperty(k, String(v));
        return;
      }
      if (typeof v === "number" && v !== 0 && !UNITLESS[k]) {
        v = v + "px";
      }
      el.style[k] = v;
    });
  }

  function applyProps(el, p) {
    Object.keys(p).forEach(function (k) {
      var v = p[k];
      if (SKIP_PROPS[k] || v === null || v === undefined || v === false) {
        return;
      }
      if (k === "className") {
        el.className = v;
      } else if (k === "style") {
        applyStyle(el, v);
      } else if (k === "tabIndex") {
        el.setAttribute("tabindex", String(v));
      } else if (k === "colSpan" || k === "rowSpan") {
        el.setAttribute(k.toLowerCase(), String(v));
      } else if (k === "htmlFor") {
        el.setAttribute("for", String(v));
      } else {
        el.setAttribute(k, v === true ? "" : String(v));
      }
    });
  }

  // Build a static subtree into a DOM node.
  function buildInto(parent, n) {
    if (n === null || n === undefined || n === false || n === true) {
      return;
    }
    if (typeof n === "string" || typeof n === "number") {
      parent.appendChild(document.createTextNode(String(n)));
      return;
    }
    if (Array.isArray(n)) {
      n.forEach(function (c) {
        buildInto(parent, c);
      });
      return;
    }
    if (!isComp(n)) {
      return;
    }
    var p = n.props || {};
    if (n.namespace !== HTML_NS) {
      // A tooltip: its child, with the label as title.
      var before = parent.childNodes.length;
      buildInto(parent, p.children);
      var el0 = parent.childNodes[before];
      if (el0 && el0.nodeType === 1) {
        if (!el0.getAttribute("title")) {
          el0.setAttribute("title", String(p.label));
        }
      } else if (el0) {
        var span = document.createElement("span");
        span.setAttribute("title", String(p.label));
        parent.replaceChild(span, el0);
        span.appendChild(el0);
      }
      return;
    }
    var el = document.createElement(String(n.type).toLowerCase());
    applyProps(el, p);
    buildInto(el, p.children);
    parent.appendChild(el);
  }

  function fillIslands(isl, tries) {
    var left = 0;
    Object.keys(isl).forEach(function (key) {
      var host = document.querySelector('#ib-preview [data-ib-island="' + key + '"]');
      if (!host) {
        left += 1;
        return;
      }
      if (host.getAttribute("data-ib-filled") === "1") {
        return;
      }
      var frag = document.createDocumentFragment();
      buildInto(frag, isl[key]);
      host.appendChild(frag);
      host.setAttribute("data-ib-filled", "1");
    });
    if (left && tries < 40) {
      requestAnimationFrame(function () {
        fillIslands(isl, tries + 1);
      });
    }
  }

  // ------------------------------------------------------------------ selection

  // The id of the row that should be selected in a panel (the grids' getRowId).
  function selKey(panel) {
    if (S.pending[panel] !== undefined) {
      return S.pending[panel];
    }
    var sel = S.sel || {};
    var which = whichOf(panel);
    if (which === "protein_group") {
      return sel.group ? String(sel.group) : null;
    }
    if (which === "peptide") {
      return isNum(sel.peptide) ? String(sel.peptide) : null;
    }
    return isNum(sel.cid) ? (sel.run || "") + ":" + sel.cid : null;
  }

  // Run fn with a panel's grid API once the grid exists (it may mount after the
  // callback that asks for it).
  function withApi(panel, fn, tries) {
    var id = GRIDS[panel];
    var api = gridApi(id);
    if (api) {
      fn(api);
      return;
    }
    if (!document.getElementById(PANELS[panel])) {
      return; // this level has no such panel
    }
    var dag = window.dash_ag_grid;
    if (document.getElementById(id) && dag && dag.getApiAsync) {
      try {
        dag.getApiAsync(id).then(function (a) {
          if (a && !a.isDestroyed()) {
            fn(a);
          }
        });
      } catch (e) {
        /* the grid is gone */
      }
      return;
    }
    tries = tries === undefined ? 30 : tries;
    if (tries > 0) {
      setTimeout(function () {
        withApi(panel, fn, tries - 1);
      }, 100);
    }
  }

  function highlight(panel, scroll) {
    withApi(panel, function (api) {
      select(panel, api, scroll);
    });
  }

  // A row is fully inside the viewport of its grid.
  function inView(api, node) {
    if (!node || node.rowTop === null || node.rowTop === undefined) {
      return false;
    }
    var range = api.getVerticalPixelRange();
    var height = node.rowHeight || ROW_HEIGHT;
    return node.rowTop >= range.top - 1 && node.rowTop + height <= range.bottom + 1;
  }

  function select(panel, api, scroll) {
    var key = selKey(panel);
    var node = key !== null ? api.getRowNode(key) : null;
    if (node && node.data) {
      if (!node.isSelected()) {
        node.setSelected(true, true);
      }
      if (scroll && panel !== "top" && !inView(api, node)) {
        api.ensureNodeVisible(node, "middle");
      }
    } else if (api.getSelectedNodes && api.getSelectedNodes().length) {
      api.deselectAll();
    }
  }

  function highlightAll(scroll) {
    ["top", "pep", "pre"].forEach(function (p) {
      highlight(p, scroll);
    });
    coverageFollow();
  }

  // ------------------------------------------------------------------ coverage
  // The coverage strip of the peptides panel (ui/coverage.py, assets/coverage.js): asked
  // for when the selected protein group or the threshold changes; a new peptide only
  // moves the outline.

  function coverageFollow() {
    if (!document.getElementById("ib-cov") || !S.view) {
      return;
    }
    var group = S.view.unit === "protein_group" ? S.sel.group || null : null;
    var key = group ? group + "|" + S.view.t : null;
    if (key !== S.covKey) {
      S.covKey = key;
      S.covMember = null;
      if (group) {
        setProps("ib-cov-req", {
          data: { group: group, member: null, peptide: S.sel.peptide, t: S.view.t, n: now() },
        });
      }
    }
    if (window.mvCoverage) {
      window.mvCoverage.select(S.sel.peptide);
    }
  }

  // A click on the bar: select that peptide in the peptides panel.
  document.addEventListener("mv-coverage-pick", function (ev) {
    var d = ev.detail || {};
    if (!onThisPage() || !S.view || !S.sel || String(d.group) !== String(S.sel.group)) {
      return;
    }
    var api = gridApi(GRIDS.pep);
    if (!api) {
      return;
    }
    var node = null;
    api.forEachNode(function (n) {
      if (!node && n.data && same(n.data.base_peptide_id, d.pep)) {
        node = n;
      }
    });
    if (node && node.data) {
      node.setSelected(true, true);
      api.ensureNodeVisible(node, "middle");
      pick("pep", node.data);
    }
  });

  // A click on another member of the group: its coverage.
  document.addEventListener("mv-coverage-member", function (ev) {
    var d = ev.detail || {};
    if (!onThisPage() || !S.view || String(d.group) !== String(S.sel.group)) {
      return;
    }
    S.covMember = d.member;
    setProps("ib-cov-req", {
      data: { group: d.group, member: d.member, peptide: S.sel.peptide, t: S.view.t, n: now() },
    });
  });

  // A child panel waits for its rows: dimmed; the preview waits for it.
  function loading(panels, on) {
    panels.forEach(function (p) {
      var el = document.getElementById(PANELS[p]);
      if (el) {
        el.classList.toggle("ib-loading", on);
      }
      S.waiting[p] = !!on && !!el;
      clearTimeout(S.loadTimers[p]);
      if (on) {
        // A safety net: the panel comes back after a few seconds whatever happens.
        S.loadTimers[p] = setTimeout(function () {
          loading([p], false);
        }, 12000);
      }
    });
    if (!on) {
      scheduleInsert();
    }
  }

  function waitingForChildren() {
    return !!(S.waiting.pep || S.waiting.pre);
  }

  function previewLoading(on) {
    var el = document.getElementById("ib-panel-prev");
    if (el) {
      el.classList.toggle("ib-loading", on);
    }
  }

  var COMPACT_FIELDS = [
    "peptidoform",
    "charge",
    "label",
    "run",
    "score",
    "q_value",
    "run_psm_q",
    "precursor_q",
    "peptide_q_value",
    "pg_q_value",
  ];

  function compact(row) {
    var out = {};
    if (!row) {
      return out;
    }
    COMPACT_FIELDS.forEach(function (k) {
      if (row[k] !== undefined) {
        out[k] = row[k];
      }
    });
    return out;
  }

  // ------------------------------------------------------------------ children

  function partKey(panel, key, decoys) {
    return panel + "|" + String(key) + "|" + (decoys ? 1 : 0);
  }

  function needKey(need) {
    return need.kind === "group"
      ? partKey("pep", need.group, need.decoys)
      : partKey("pre", need.peptide, need.decoys);
  }

  // The first row whose `key` is `wanted`, else the first row (browser._pick).
  function pickBy(rows, key, wanted) {
    if (!rows || !rows.length) {
      return null;
    }
    if (isNum(wanted)) {
      for (var i = 0; i < rows.length; i++) {
        if (same(rows[i][key], wanted)) {
          return rows[i];
        }
      }
    }
    return rows[0];
  }

  // The wanted precursor (run and candidate id), else the first row (browser._pick_precursor).
  function pickPrecursor(rows, run, cid) {
    if (!rows || !rows.length) {
      return null;
    }
    if (isNum(cid)) {
      for (var i = 0; i < rows.length; i++) {
        if (same(rows[i].candidate_id, cid) && (rows[i].run || "") === (run || "")) {
          return rows[i];
        }
      }
    }
    return rows[0];
  }

  function cachePayload(p) {
    var decoys = !!p.decoys;
    if (p.pep && !p.pep.error) {
      C.set(partKey("pep", p.pep.key, decoys), p.pep);
    }
    if (p.pre && !p.pre.error) {
      C.set(partKey("pre", p.pre.key, decoys), p.pre);
    }
  }

  // The answer to a request from the parts kept here (the server's rule: the wanted
  // peptide and precursor when they are rows of their parent, else the first rows).
  function resolveFromCache(need, cache) {
    cache = cache || C;
    var decoys = !!need.decoys;
    var out = { kind: need.kind, n: need.n, decoys: decoys, found: true };
    var peptide = need.peptide;
    var run = need.run || "";
    var cid = need.cid;
    if (need.kind === "group") {
      var pep = cache.get(partKey("pep", need.group, decoys));
      if (!pep) {
        return null;
      }
      out.pep = pep;
      var row = pickBy(pep.rows, "base_peptide_id", peptide);
      if (!row) {
        out.found = !!pep.error;
        out.pre = null;
        out.prec = null;
        out.sel = { group: need.group, peptide: null, run: "", cid: null };
        return out;
      }
      if (!same(row.base_peptide_id, peptide)) {
        run = "";
        cid = null;
      }
      peptide = row.base_peptide_id;
    }
    if (!isNum(peptide)) {
      return null;
    }
    var pre = cache.get(partKey("pre", peptide, decoys));
    if (!pre) {
      return null;
    }
    out.pre = pre;
    if (!pre.rows.length && !pre.error) {
      out.found = false;
    }
    var prec = pickPrecursor(pre.rows, run, cid);
    out.sel = {
      group: need.group || null,
      peptide: peptide,
      run: prec ? prec.run || "" : "",
      cid: prec ? prec.candidate_id : null,
    };
    out.prec = prec ? { run: prec.run || "", cid: prec.candidate_id, row: compact(prec) } : null;
    return out;
  }

  // Rows that pass the header threshold on the panel's mark column (the marks' test:
  // a target with an engine q at most t). A count of engine values, not a new q.
  function passCount(rows, field, t) {
    var n = 0;
    (rows || []).forEach(function (r) {
      var q = r[field];
      if (r.label !== "decoy" && isNum(q) && Number(q) <= Number(t)) {
        n += 1;
      }
    });
    return n;
  }

  function updateCount(panel) {
    var part = S.parts[panel];
    if (!part || !S.view) {
      return;
    }
    var loaded = part.rows.length;
    var t = S.view.t;
    var text;
    var tip;
    if (part.error) {
      text = "-";
      tip = "The data layer refused this table.";
    } else if (loaded < part.total) {
      text = fmtInt(loaded) + " of " + fmtInt(part.total) + " loaded";
      tip = "The rows are loading in blocks of 2,000 (best score first).";
    } else {
      var pass = passCount(part.rows, part.mark, t);
      text = fmtInt(pass) + " of " + fmtInt(part.total);
      tip =
        fmtInt(pass) +
        " of " +
        fmtInt(part.total) +
        " " +
        NOUN[panel] +
        " pass " +
        part.mark +
        " ≤ " +
        stopLabel(t) +
        " (targets; the validation marks' test). The table lists every row of its parent, at any q.";
    }
    setProps("ib-" + panel + "-count", { children: text });
    setProps("ib-" + panel + "-countbox", { title: tip });
  }

  function applyPart(panel, part) {
    if (!document.getElementById(GRIDS[panel])) {
      return;
    }
    S.parts[panel] = part;
    setProps(GRIDS[panel], { rowData: part.rows });
    setProps("ib-" + panel + "-subject", { children: part.subject });
    setProps("ib-" + panel + "-help", { label: part.help });
    setProps("ib-" + panel + "-note", { children: part.note });
    updateCount(panel);
    // After the grid has painted its rows.
    requestAnimationFrame(function () {
      highlight(panel, true);
      loading([panel], false);
    });
  }

  function emptyPart(panel) {
    if (!document.getElementById(GRIDS[panel])) {
      return;
    }
    S.parts[panel] = null;
    setProps(GRIDS[panel], { rowData: [] });
    setProps("ib-" + panel + "-subject", { children: "" });
    setProps("ib-" + panel + "-count", { children: "0" });
    setProps("ib-" + panel + "-note", { children: "" });
    loading([panel], false);
  }

  // Ask for the child panels of a selection (or take them from the cache).
  function requestChildren(kind, sel, peptidoform) {
    if (!S.view) {
      return;
    }
    var decoys = !!S.view.decoys;
    S.nonce += 1;
    var need = {
      kind: kind,
      group: sel.group || null,
      peptide: isNum(sel.peptide) ? sel.peptide : null,
      run: sel.run || "",
      cid: isNum(sel.cid) ? sel.cid : null,
      decoys: decoys,
      peptidoform: peptidoform || null,
      n: S.nonce,
    };
    S.need = need;
    var hit = resolveFromCache(need);
    if (hit) {
      applyChildren(hit);
      return;
    }
    if (kind === "group") {
      var pep = C.get(partKey("pep", need.group, decoys));
      if (pep && pep.rows.length) {
        // The peptides are kept: show them, ask for the chosen peptide's precursors.
        var row = pickBy(pep.rows, "base_peptide_id", need.peptide);
        var keep = same(row.base_peptide_id, need.peptide);
        applyPart("pep", pep);
        need = Object.assign({}, need, {
          kind: "peptide",
          peptide: row.base_peptide_id,
          run: keep ? need.run : "",
          cid: keep ? need.cid : null,
          peptidoform: row.peptidoform || null,
        });
        S.need = need;
        loading(["pre"], true);
        sendNeed(need);
        return;
      }
    }
    loading(kind === "group" ? ["pep", "pre"] : ["pre"], true);
    sendNeed(need);
  }

  // Send a children request; once again when no answer has come after a while (a request
  // the network dropped would otherwise leave the panels waiting).
  function sendNeed(need) {
    setProps("ib-need", { data: need });
    clearTimeout(S.needTimer);
    S.needTimer = setTimeout(function () {
      if (S.need === need && S.appliedN !== need.n && root()) {
        setProps("ib-need", { data: Object.assign({}, need, { retry: 1 }) });
      }
    }, 6000);
  }

  function linkText(kind, need) {
    if (kind === "group") {
      return "The protein group in the link (" + need.group + ") has no rows in this result set" +
        (need.decoys ? "" : " (decoys are hidden)") + ". The first row is selected.";
    }
    return "The peptide in the link (base_peptide_id " + need.peptide + ") has no rows in this result set" +
      (need.decoys ? "" : " (decoys are hidden)") + ". The first row is selected.";
  }

  // Show a children answer (from the server or the cache) and follow its selection.
  function applyChildren(p) {
    if (!S.view || !root()) {
      return;
    }
    S.appliedN = p.n;
    if (p.pep) {
      applyPart("pep", p.pep);
    }
    if (p.pre) {
      applyPart("pre", p.pre);
    } else if (p.pre === null) {
      emptyPart("pre");
    }
    if (p.found === false && S.addrSel) {
      var what = p.kind === "group" && (!p.pep || !p.pep.rows.length) ? "group" : "peptide";
      fallbackToFirst(linkText(what, S.need || {}));
      return;
    }
    var sel = Object.assign({}, S.sel, p.sel || {});
    if (S.view.unit !== "protein_group") {
      sel.group = null;
    }
    S.sel = sel;
    S.pending = {};
    highlightAll(true);
    writeAddress();
    wantPreview(sel.run, sel.cid, p.prec ? p.prec.row : null, keyDelay());
    ["pep", "pre"].forEach(function (panel) {
      var part = S.parts[panel];
      if (part && !part.error && part.rows.length < part.total) {
        requestMore(panel);
      }
    });
    schedulePrefetch();
  }

  // The next block of a long child table.
  function requestMore(panel) {
    var part = S.parts[panel];
    if (!part || !S.view) {
      return;
    }
    var need = {
      kind: "more",
      unit: panel === "pep" ? "peptide" : "precursor",
      group: panel === "pep" ? part.key : null,
      peptide: panel === "pre" ? part.key : null,
      offset: part.rows.length,
      decoys: !!S.view.decoys,
      n: S.nonce,
    };
    setTimeout(function () {
      if (S.nonce === need.n && S.parts[panel] === part) {
        setProps("ib-need", { data: need });
      }
    }, 120);
  }

  function applyMore(m, n) {
    if (!m || n !== S.nonce || !S.view) {
      return;
    }
    var panel = m.unit === "peptide" ? "pep" : "pre";
    var part = S.parts[panel];
    if (!part || String(part.key) !== String(m.key) || part.rows.length !== m.offset || m.error) {
      return;
    }
    var rows = part.rows.concat(m.rows || []);
    var np = Object.assign({}, part, { rows: rows, total: m.total });
    np.note =
      rows.length < np.total
        ? fmtInt(rows.length) + " of " + fmtInt(np.total) + " loaded · any q"
        : "any q";
    C.set(partKey(panel, part.key, S.view.decoys), np);
    S.parts[panel] = np;
    setProps(GRIDS[panel], { rowData: rows });
    setProps("ib-" + panel + "-note", { children: np.note });
    updateCount(panel);
    if (rows.length < np.total && (m.rows || []).length) {
      requestMore(panel);
    }
  }

  // ------------------------------------------------------------------ prefetch

  function needForRow(panel, row) {
    if (!row || !S.view) {
      return null;
    }
    var decoys = !!S.view.decoys;
    var which = whichOf(panel);
    if (which === "protein_group") {
      return {
        kind: "group",
        group: row.protein_group,
        peptide: row.base_peptide_id,
        run: row.run || "",
        cid: row.candidate_id,
        decoys: decoys,
        peptidoform: row.peptidoform || null,
      };
    }
    if (which === "peptide") {
      return {
        kind: "peptide",
        group: S.view.unit === "protein_group" ? S.sel.group || null : null,
        peptide: row.base_peptide_id,
        run: row.run || "",
        cid: row.candidate_id,
        decoys: decoys,
        peptidoform: row.peptidoform || null,
      };
    }
    return null;
  }

  function schedulePrefetch() {
    clearTimeout(S.pfTimer);
    S.pfTimer = setTimeout(prefetch, PREFETCH_DELAY);
  }

  // Ask ahead for the children of the next row in the direction of travel.
  function prefetch() {
    if (!S.view || !root() || S.view.unit === "precursor") {
      return;
    }
    if (waitingForChildren() || now() - S.lastKey < PREFETCH_DELAY) {
      schedulePrefetch();
      return;
    }
    var panel = S.lastPanel === "pep" && S.view.unit === "protein_group" ? "pep" : "top";
    var api = gridApi(GRIDS[panel]);
    var key = selKey(panel);
    var node = api && key !== null ? api.getRowNode(key) : null;
    if (!node || node.rowIndex === null || node.rowIndex === undefined) {
      return;
    }
    var next = api.getDisplayedRowAtIndex(node.rowIndex + (S.dir < 0 ? -1 : 1));
    var need = next && next.data ? needForRow(panel, next.data) : null;
    if (!need || resolveFromCache(need)) {
      return;
    }
    S.pfN += 1;
    setProps("ib-prefetch", { data: { needs: [need], n: S.pfN } });
  }

  // ------------------------------------------------------------------ preview

  function pvKey(run, cid, t) {
    return (run || "") + ":" + cid + ":" + t;
  }

  function keyDelay() {
    return now() - S.lastKey < 600 ? PREVIEW_KEY_DELAY : 0;
  }

  // The preview should show this precursor: ask for its card (now, or after `delay`
  // while the keys move the selection) unless it is kept.
  function wantPreview(run, cid, row, delay) {
    if (!S.view || !document.getElementById("ib-preview")) {
      return;
    }
    clearTimeout(PV.timer);
    if (!isNum(cid)) {
      PV.want = null;
      showNoPreview();
      return;
    }
    var key = pvKey(run, cid, S.view.t);
    if (PV.want && PV.want.key === key) {
      if (PV.shown !== key) {
        scheduleInsert();
      }
      return;
    }
    var w = { run: run || "", cid: cid, row: row || {}, key: key, t: S.view.t };
    PV.want = w;
    if (PV.shown === key) {
      previewLoading(false);
      return;
    }
    previewLoading(true);
    if (PV.cards.has(key)) {
      scheduleInsert();
      return;
    }
    var ask = function () {
      if (PV.want !== w) {
        return;
      }
      PV.n += 1;
      setProps("ib-prec", { data: { run: w.run, cid: w.cid, row: compact(w.row), t: w.t, n: PV.n } });
    };
    if (delay) {
      PV.timer = setTimeout(ask, delay);
    } else {
      ask();
    }
  }

  function idle(fn) {
    if (window.requestIdleCallback) {
      window.requestIdleCallback(fn, { timeout: 250 });
    } else {
      setTimeout(fn, 16);
    }
  }

  // Insert the wanted card once the child panels have their rows and no key is held.
  function scheduleInsert() {
    clearTimeout(PV.insTimer);
    var w = PV.want;
    if (!w || !PV.cards.has(w.key) || PV.shown === w.key || waitingForChildren()) {
      return;
    }
    var wait = Math.max(0, S.lastKey + KEY_QUIET - now());
    PV.insTimer = setTimeout(function () {
      idle(function () {
        if (PV.want !== w || waitingForChildren()) {
          return;
        }
        if (S.lastKey + KEY_QUIET > now()) {
          scheduleInsert();
          return;
        }
        insertCard(PV.cards.get(w.key), w.key);
      });
    }, wait);
  }

  function collapseTree(tree) {
    if (!isComp(tree)) {
      return tree;
    }
    var kids = tree.props && tree.props.children;
    if (!Array.isArray(kids) || kids.length < 2) {
      return tree;
    }
    var cls = String(tree.props.className || "") + " ib-pv-collapsed";
    return Object.assign({}, tree, {
      props: Object.assign({}, tree.props, { children: [kids[0]], className: cls }),
    });
  }

  // The card's figures, each replaced by a host of the figure's height; they are mounted
  // one by one afterwards, so no single task draws the card and both figures.
  function deferGraphs(n, out) {
    if (Array.isArray(n)) {
      return n.map(function (c) {
        return deferGraphs(c, out);
      });
    }
    if (!isComp(n)) {
      return n;
    }
    var p = n.props || {};
    if (n.namespace === "dash_core_components" && n.type === "Graph") {
      PV.seq += 1;
      var id = "ib-pvg-" + PV.seq;
      var fig = p.figure || {};
      var height = fig.layout && Number(fig.layout.height);
      var hp = { id: id, className: "ib-pv-graphhost" };
      if (height > 0) {
        hp.style = { minHeight: height + "px" };
      }
      out.push({ id: id, node: n });
      return { namespace: HTML_NS, type: "Div", props: hp };
    }
    if (p.children === undefined || p.children === null) {
      return n;
    }
    return Object.assign({}, n, {
      props: Object.assign({}, p, { children: deferGraphs(p.children, out) }),
    });
  }

  function insertGraphs(graphs, key, i) {
    if (i >= graphs.length) {
      return;
    }
    idle(function () {
      if (PV.shown !== key || !root()) {
        return; // another card is shown
      }
      var wait = S.lastKey + KEY_QUIET - now();
      if (wait > 0) {
        setTimeout(function () {
          insertGraphs(graphs, key, i);
        }, wait);
        return;
      }
      setProps(graphs[i].id, { children: graphs[i].node });
      insertGraphs(graphs, key, i + 1);
    });
  }

  function insertCard(data, key) {
    if (!data || !root()) {
      return;
    }
    var isl = {};
    var graphs = [];
    var tree = data.card;
    if (PV.collapsed) {
      tree = collapseTree(tree);
    }
    tree = deferGraphs(islandize(tree, isl), graphs);
    setProps("ib-preview", { children: tree });
    PV.shown = key;
    previewLoading(false);
    fillIslands(isl, 0);
    insertGraphs(graphs, key, 0);
  }

  function showNoPreview() {
    PV.shown = "none";
    setProps("ib-preview", {
      children: {
        namespace: HTML_NS,
        type: "Div",
        props: {
          className: "ib-preview-empty ib-preview-none",
          children: "No precursor is selected. Select a row above: its XIC, spectrum and ion table appear here.",
        },
      },
    });
    previewLoading(false);
  }

  function setCollapsedAttr() {
    document.documentElement.setAttribute("data-ib-preview", PV.collapsed ? "collapsed" : "open");
  }
  setCollapsedAttr();

  function toggleCollapsed() {
    PV.collapsed = !PV.collapsed;
    try {
      window.localStorage.setItem(COLLAPSE_KEY, PV.collapsed ? "1" : "0");
    } catch (e) {
      /* a viewer convenience only */
    }
    setCollapsedAttr();
    var key = PV.shown;
    if (key && PV.cards.has(key)) {
      insertCard(PV.cards.get(key), key);
    }
    // The grids get (or give back) the room.
    setTimeout(function () {
      ["top", "pep", "pre"].forEach(function (p) {
        var api = gridApi(GRIDS[p]);
        if (api) {
          updateRange(api, p);
        }
      });
    }, 50);
  }

  // ------------------------------------------------------------------ notices

  function showNotice(text) {
    setProps("ib-notice-text", { children: text });
    setProps("ib-notice", { className: "ib-notice" });
  }

  function hideNotice() {
    setProps("ib-notice", { className: "ib-notice ib-hidden" });
  }

  // A link names a row this table lacks: select the first row instead and say so.
  function fallbackToFirst(text) {
    S.addrSel = false;
    showNotice(text);
    withApi("top", function (api) {
      var n0 = api.getDisplayedRowAtIndex(0);
      if (n0 && n0.data) {
        pick("top", n0.data, { force: true });
      }
    });
  }

  // ------------------------------------------------------------------ picks

  // Select a row of a panel: the panels below follow.
  function pick(panel, row, opts) {
    opts = opts || {};
    if (!row || !S.view || !onThisPage()) {
      return;
    }
    delete S.pending[panel];
    var which = whichOf(panel);
    if (!opts.force && selKey(panel) === row._key) {
      return;
    }
    if (!opts.keepAddress) {
      S.addrSel = false;
    }
    S.lastPanel = panel;
    var run = row.run || "";
    var cid = row.candidate_id;
    var old = S.sel || {};
    var sel;
    // The row shows its precursor: its preview is asked for together with the children
    // (which confirm it, or name the precursor to show instead).
    var delay = opts.key ? PREVIEW_KEY_DELAY : 0;
    if (which === "protein_group") {
      sel = { group: row.protein_group, peptide: row.base_peptide_id, run: run, cid: cid };
      S.sel = sel;
      wantPreview(run, cid, row, delay);
      requestChildren("group", sel, row.peptidoform);
    } else if (which === "peptide") {
      sel = {
        group: S.view.unit === "protein_group" ? old.group : null,
        peptide: row.base_peptide_id,
        run: run,
        cid: cid,
      };
      S.sel = sel;
      wantPreview(run, cid, row, delay);
      requestChildren("peptide", sel, row.peptidoform);
    } else {
      sel = Object.assign({}, old, { run: run, cid: cid });
      S.sel = sel;
      wantPreview(run, cid, row, delay);
    }
    highlightAll(false);
    writeAddress();
    if (panel === "top") {
      checkInView();
    }
  }

  // A focus move by the keys: the selection moves at once, the panels follow after a
  // pause.
  function schedulePick(panel, index, api) {
    if (index === null || index === undefined || index < 0) {
      return;
    }
    clearTimeout(S.timers[panel]);
    var node = api.getDisplayedRowAtIndex(index);
    if (node && node.data) {
      S.pending[panel] = node.data._key;
      node.setSelected(true, true);
      delete S.focusPending[panel];
    } else {
      // A row of a block still loading: picked when it arrives (mvbModelUpdated).
      S.focusPending[panel] = index;
      return;
    }
    S.timers[panel] = setTimeout(function () {
      if (!api || api.isDestroyed()) {
        return;
      }
      var n = api.getDisplayedRowAtIndex(index);
      delete S.pending[panel];
      if (n && n.data) {
        pick(panel, n.data, { key: true });
      }
    }, DEBOUNCE_MS);
  }

  function clearSelection() {
    S.sel = {};
    S.pending = {};
    ["pep", "pre"].forEach(function (p) {
      if (document.getElementById(GRIDS[p])) {
        emptyPart(p);
      }
    });
    PV.want = null;
    showNoPreview();
    writeAddress();
    checkInView();
  }

  function setThreshold(t) {
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
      if (p !== "top") {
        // The validation header names the threshold.
        var col = api.getColumn("_valid");
        var def = col && col.getColDef();
        var tpl = def && def.cellRendererParams && def.cellRendererParams.tipTemplate;
        if (tpl) {
          def.headerTooltip = tpl.replace("{t}", stopLabel(t));
          api.refreshHeader();
        }
      }
      api.refreshCells({ force: true });
    });
    updateCount("pep");
    updateCount("pre");
  }

  function updateRange(api, panel) {
    if (panel && panel !== "top") {
      return;
    }
    var el = document.getElementById("ib-range");
    if (!el || !api) {
      return;
    }
    var n = api.getDisplayedRowCount();
    var known = api.isLastRowIndexKnown ? api.isLastRowIndexKnown() : true;
    if (!known) {
      el.textContent = "loading";
      return;
    }
    if (!n) {
      el.textContent = "no rows";
      return;
    }
    var range = api.getVerticalPixelRange();
    var height = api.getGridOption("rowHeight") || ROW_HEIGHT;
    var first = Math.max(0, Math.floor((range.top + 2) / height));
    var last = Math.min(n - 1, Math.max(first, Math.floor((range.bottom - 2) / height)));
    var sort = currentSort();
    el.textContent =
      fmtInt(first + 1) + "–" + fmtInt(last + 1) + " of " + fmtInt(n) + " · " + sort.col + (sort.desc ? " ↓" : " ↑");
  }

  // ------------------------------------------------------------------ selected row in view

  // The title bar offers "show the selected row" when the first table's selected row is
  // not on screen.
  function checkInView() {
    var btn = document.getElementById("ib-locate");
    var api = gridApi(GRIDS.top);
    if (!btn || !api) {
      return;
    }
    var key = selKey("top");
    var node = key !== null ? api.getRowNode(key) : null;
    var off = key !== null && !(node && node.data && inView(api, node));
    if (off !== S.locShown) {
      S.locShown = off;
      btn.classList.toggle("ib-hidden", !off);
    }
  }

  function showSelected() {
    var api = gridApi(GRIDS.top);
    var key = selKey("top");
    if (!api || key === null) {
      return;
    }
    var node = api.getRowNode(key);
    if (node && node.data && node.rowIndex !== null && node.rowIndex !== undefined) {
      api.ensureIndexVisible(node.rowIndex, "middle");
      return;
    }
    var sort = currentSort();
    S.locN += 1;
    setProps("ib-locate-req", { data: { key: key, sort: [sort.col, sort.desc], n: S.locN } });
  }

  // ------------------------------------------------------------------ narrow windows

  // A grid whose columns do not fit its width hides the columns that allow it
  // (context.hideOrder, the highest first) until they fit; once per grid and page (and
  // after Defaults), so a column the user shows again stays. The Columns menu follows.
  function colWidth(c) {
    var d = c.getColDef() || {};
    // A flexible column can shrink to its minimum width (or the width it asks to keep).
    if (d.flex) {
      var min = (c.getMinWidth && c.getMinWidth()) || d.minWidth || c.getActualWidth();
      return Math.max(min, (d.context && d.context.fitWidth) || 0);
    }
    return c.getActualWidth();
  }

  // Twice per grid and page: when the page starts and when the grid shows its first rows
  // (a vertical scroll bar can take width then).
  function applyNarrow(panel, api, force, phase) {
    phase = phase || "start";
    var done = S.page + ":" + phase;
    if (!api || (!force && S.narrowDone[panel + ":" + phase] === done)) {
      return;
    }
    var el = document.getElementById(GRIDS[panel]);
    var viewport = el && el.querySelector(".ag-center-cols-viewport");
    var avail = viewport ? viewport.clientWidth : 0;
    if (!avail) {
      return; // not laid out yet; the next call tries again
    }
    S.narrowDone[panel + ":" + phase] = done;
    var cols = api.getAllDisplayedColumns();
    var total = 0;
    cols.forEach(function (c) {
      if (!c.getPinned()) {
        total += colWidth(c);
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
      total -= colWidth(c);
    }
    if (hide.length) {
      api.setColumnsVisible(hide, false);
      syncMenu(panel, api);
    }
    if (panel === "top" && window.innerWidth < 1280) {
      setProps("ib-search", { placeholder: "Search" });
    }
  }

  function shownColumns(api) {
    return api
      .getColumns()
      .filter(function (c) {
        var d = c.getColDef() || {};
        return c.isVisible() && !FIXED[c.getColId()] && !d.lockVisible;
      })
      .map(function (c) {
        return c.getColId();
      });
  }

  function syncMenu(panel, api) {
    setProps(MENUS[panel], { value: shownColumns(api) });
  }

  // ------------------------------------------------------------------ the page's start

  // The first table shows its first block: ask for the child panels and the preview of
  // the page's selection, and find the selected row when a link names one further down.
  function start() {
    if (S.started || !S.view || !root()) {
      return;
    }
    S.started = true;
    clearTimeout(S.startTimer);
    var unit = S.view.unit;
    var sel = S.sel || {};
    if (unit === "protein_group") {
      if (sel.group) {
        requestChildren("group", sel, null);
      } else {
        emptyPart("pep");
        emptyPart("pre");
      }
    } else if (unit === "peptide") {
      if (isNum(sel.peptide)) {
        requestChildren("peptide", sel, null);
      } else {
        emptyPart("pre");
      }
    }
    wantPreview(sel.run, sel.cid, S.prec0 && S.prec0.row, 0);
    ["pep", "pre"].forEach(function (panel) {
      withApi(panel, function (a) {
        applyNarrow(panel, a);
      });
    });
    var api = gridApi(GRIDS.top);
    if (api) {
      applyNarrow("top", api);
      var key = selKey("top");
      var node = key !== null ? api.getRowNode(key) : null;
      if (key !== null && S.addrSel && !(node && node.data && inView(api, node))) {
        // A link names a row further down: show it.
        showSelected();
      }
    }
    checkInView();
  }

  function addressSelects(built) {
    var q = built || {};
    return !!(q.group || isNum(q.peptide) || isNum(q.cid));
  }

  function open(row) {
    if (row && row._href) {
      go(row._href);
    }
  }

  function selectedRow(panel, api) {
    var nodes = api && api.getSelectedNodes ? api.getSelectedNodes() : [];
    return nodes && nodes.length && nodes[0].data ? nodes[0].data : null;
  }

  // ------------------------------------------------------------------ grid functions

  var F = {
    // The grids follow the Mantine theme: every colour is a CSS variable that
    // browser.css sets for the light and the dark scheme.
    mvbTheme: function (base) {
      return base.withParams({
        fontFamily: "inherit",
        fontSize: 12.5,
        headerFontSize: 12,
        headerFontWeight: 600,
        backgroundColor: "var(--ib-grid-bg)",
        foregroundColor: "var(--ib-grid-fg)",
        textColor: "var(--ib-grid-fg)",
        subtleTextColor: "var(--ib-grid-dim)",
        borderColor: "var(--ib-grid-border)",
        chromeBackgroundColor: "var(--ib-grid-chrome)",
        headerBackgroundColor: "var(--ib-grid-header)",
        headerTextColor: "var(--ib-grid-header-fg)",
        rowHoverColor: "var(--ib-grid-hover)",
        selectedRowBackgroundColor: "var(--ib-grid-selected)",
        accentColor: "var(--ib-grid-accent)",
        wrapperBorder: false,
        wrapperBorderRadius: 0,
        rowBorder: { style: "solid", width: 1, color: "var(--ib-grid-rowline)" },
        columnBorder: false,
        headerRowBorder: { style: "solid", width: 1, color: "var(--ib-grid-border)" },
        pinnedColumnBorder: { style: "solid", width: 1, color: "var(--ib-grid-border)" },
        spacing: 4,
        cellHorizontalPadding: 8,
        rowHeight: ROW_HEIGHT,
        headerHeight: 28,
        iconSize: 13,
        headerColumnResizeHandleColor: "transparent",
        rangeSelectionBorderColor: "var(--ib-grid-accent)",
        browserColorScheme: "inherit",
        tooltipBackgroundColor: "var(--ib-tip-bg)",
        tooltipTextColor: "var(--ib-tip-fg)",
        tooltipBorder: false,
      });
    },

    mvbQuantityTip: function (params) {
      if (!params || !params.data) {
        return "";
      }
      var v = params.value;
      if (v === null || v === undefined) {
        var state = params.data.quant_state;
        return "No quantity (" + (QUANT_WORDS[state] || state || "no quant table") + "); never 0";
      }
      return Number(v).toLocaleString("en-US", { maximumFractionDigits: 2 }) + " (intensity x s)";
    },

    mvbGroupTip: function (params) {
      var members = String((params && params.value) || "").split(";").filter(Boolean);
      return members.length > 1 ? members.length + " members: " + members.join(", ") : undefined;
    },

    // Arrow keys (navigateToNextCell): AG Grid moves the focus and mvbFocused moves the
    // selection with it. The focus stays in the body: above the first row is the
    // header (and the infinite model would read row -1 as a row of its first block).
    mvbNavigate: function (params, panel) {
      var next = params.nextCellPosition;
      var prev = params.previousCellPosition;
      S.lastKey = now();
      S.lastPanel = panel;
      if (!next) {
        return next;
      }
      if (next.rowIndex === null || next.rowIndex === undefined || next.rowIndex < 0 || next.rowPinned) {
        return prev || null;
      }
      if (prev && next.rowIndex !== prev.rowIndex) {
        S.dir = next.rowIndex > prev.rowIndex ? 1 : -1;
      }
      return next;
    },

    // Every focus move the user makes (arrows, page keys, Home, End, a click) moves the
    // selection to the focused row, so Enter and the panels below follow the same row.
    mvbFocused: function (params, panel) {
      if (!params || params.rowIndex === null || params.rowIndex === undefined || params.rowIndex < 0) {
        return;
      }
      if (params.rowPinned || params.floating) {
        return;
      }
      var el = document.getElementById(GRIDS[panel]);
      if (!el || !el.contains(document.activeElement)) {
        return; // a focus the grid restored by itself
      }
      var api = params.api;
      var node = api.getDisplayedRowAtIndex(params.rowIndex);
      if (node && node.data && node.data._key === selKey(panel)) {
        return;
      }
      schedulePick(panel, params.rowIndex, api);
    },

    // Space would select the focused row without the panels below.
    mvbSuppressKey: function (params) {
      var ev = params && params.event;
      return !!(ev && (ev.key === " " || ev.code === "Space"));
    },

    mvbRowClicked: function (params, panel) {
      var row = params && params.data;
      if (!row) {
        return;
      }
      var target = params.event && params.event.target;
      if (target && target.closest && target.closest(".ib-open")) {
        return;
      }
      clearTimeout(S.timers[panel]);
      delete S.focusPending[panel];
      pick(panel, row);
    },

    mvbRowDoubleClicked: function (params) {
      open(params && params.data);
    },

    // Enter opens the selected row (the highlighted one, which the panels show).
    mvbKeyDown: function (params, panel) {
      var ev = params && params.event;
      if (!ev) {
        return;
      }
      if (NAV_KEYS[ev.key]) {
        S.lastKey = now();
        S.lastPanel = panel;
        if (ev.key === "PageDown" || ev.key === "End") {
          S.dir = 1;
        } else if (ev.key === "PageUp" || ev.key === "Home") {
          S.dir = -1;
        }
      }
      if (ev.key === "Enter") {
        ev.preventDefault();
        open(selectedRow(panel, params.api) || params.data);
      } else if (ev.key === " ") {
        ev.preventDefault();
      }
    },

    mvbModelUpdated: function (params, panel) {
      var api = params && params.api;
      if (!api) {
        return;
      }
      if (panel !== "top") {
        return;
      }
      select("top", api, false);
      updateRange(api, "top");
      var row0 = api.getDisplayedRowAtIndex(0);
      var ready = (row0 && row0.data) || api.isLastRowIndexKnown();
      if (ready && !S.started && S.page) {
        requestAnimationFrame(start);
      }
      if (S.sortPending && row0 && row0.data) {
        S.sortPending = false;
        afterSort(api);
      }
      var fp = S.focusPending.top;
      if (fp !== undefined) {
        var n = api.getDisplayedRowAtIndex(fp);
        if (n && n.data) {
          delete S.focusPending.top;
          schedulePick("top", fp, api);
        }
      }
      checkInView();
    },

    // New rows of a child table: select the selected row again (its rows came without
    // a selection) and bring it into view.
    mvbRowDataUpdated: function (params, panel) {
      var api = params && params.api;
      if (api && panel !== "top") {
        setTimeout(function () {
          if (!api.isDestroyed()) {
            select(panel, api, true);
          }
        }, 0);
      }
    },

    mvbFirstData: function (params, panel) {
      var api = params && params.api;
      if (api) {
        applyNarrow(panel, api, false, "data");
      }
      if (api && panel !== "top") {
        setTimeout(function () {
          if (!api.isDestroyed()) {
            select(panel, api, true);
          }
        }, 0);
      }
    },

    // New column definitions of the first table (another q column): the chosen q
    // column goes next to the table's own q column, not to the far right.
    mvbColumnsLoaded: function (params, panel) {
      var api = params && params.api;
      if (!api || panel !== "top" || !S.view || !S.view.q) {
        return;
      }
      setTimeout(function () {
        if (api.isDestroyed()) {
          return;
        }
        var cols = api.getAllGridColumns().map(function (c) {
          return c.getColId();
        });
        var active = S.view.q;
        var anchor = UNIT_Q[S.view.unit];
        if (S.view.unit === "precursor") {
          ["charge", "run", "is_transferred"].forEach(function (c) {
            if (cols.indexOf(c) >= 0) {
              anchor = c;
            }
          });
        }
        var ia = cols.indexOf(active);
        var iu = cols.indexOf(anchor);
        if (ia < 0 || iu < 0 || ia === iu + 1) {
          return;
        }
        api.moveColumns([active], ia > iu ? iu + 1 : iu);
      }, 0);
    },

    mvbSortChanged: function (params, panel) {
      if (panel === "top") {
        S.sortPending = true;
        writeAddress();
        updateRange(params && params.api, "top");
      }
    },
  };
  window.dashAgGridFunctions = Object.assign(window.dashAgGridFunctions || {}, F);

  // After a new sort of the first table: keep the selection when its row is in the new
  // first block (and show it), else select the first row.
  function afterSort(api) {
    var key = selKey("top");
    var node = key !== null ? api.getRowNode(key) : null;
    if (node && node.data && node.rowIndex !== null && node.rowIndex < 100) {
      if (!inView(api, node)) {
        api.ensureIndexVisible(node.rowIndex, "middle");
      }
      return;
    }
    var row0 = api.getDisplayedRowAtIndex(0);
    if (row0 && row0.data) {
      pick("top", row0.data, { force: true });
    }
  }

  // ------------------------------------------------------------------ callbacks

  function triggered() {
    var ctx = window.dash_clientside.callback_context;
    return ctx && ctx.triggered && ctx.triggered.length ? ctx.triggered[0] : null;
  }

  // A newly built page: reset the page state (the kept answers stay).
  function init(page, view, built) {
    if (S.page === page) {
      return false;
    }
    S.page = page;
    S.synced = null;
    S.view = view;
    S.sel = {};
    S.built = built || null;
    S.written = null;
    S.pending = {};
    S.focusPending = {};
    S.parts = {};
    S.need = null;
    S.appliedN = null;
    S.decoysChanged = false;
    S.tChanged = false;
    S.waiting = {};
    S.started = false;
    S.sortPending = false;
    S.locShown = false;
    S.covKey = null;
    S.covMember = null;
    clearTimeout(S.startTimer);
    clearTimeout(S.pfTimer);
    PV.want = null;
    PV.shown = null;
    clearTimeout(PV.timer);
    clearTimeout(PV.insTimer);
    return true;
  }

  window.dash_clientside = Object.assign({}, window.dash_clientside, {
    mvb: {
      // The view from the controls and the header threshold.
      filters: function (q, search, charge, protein, mod, quant, run, decoys, t, prev) {
        if (!prev) {
          return NO();
        }
        // The control shows the default column when none is chosen; while it still
        // shows the old default (another control changed the default, as a run does),
        // the view keeps the default.
        var chosen = q || "";
        if (!prev.q && chosen === defaultQ(prev)) {
          chosen = "";
        }
        var next = Object.assign({}, prev, {
          q: chosen,
          search: (search || "").trim(),
          charge: charge && charge !== "any" ? String(charge) : "",
          protein: (protein || "").trim(),
          mod: (mod || "").trim(),
          quant: quant && quant !== "any" ? quant : "",
          run: run && run !== "all" ? run : "",
          decoys: !!decoys,
          t: typeof t === "number" && t > 0 && t < 1 ? t : prev.t,
        });
        // The default column, chosen explicitly, is the default.
        if (next.q === defaultQ(next)) {
          next.q = "";
        }
        return JSON.stringify(next) === JSON.stringify(prev) ? NO() : next;
      },

      // A new view: the first table asks again from row 0 (its answer brings the total,
      // the description and the first block, which keeps or moves the selection) and
      // the address follows. The first call of a page writes the address and brings the
      // header threshold to the view's (a link with t= sets it).
      sync: function (view, selected, page, built) {
        if (!view) {
          return NO();
        }
        init(page, view, built);
        var prev = S.view;
        S.view = view;
        if (S.synced !== page) {
          // The first call of this page (the page's own view, not a change).
          S.synced = page;
          writeAddress();
          if (selected !== null && selected !== undefined && parseFloat(selected) !== view.t) {
            setProps("q-select", { value: String(view.t) });
          }
          return NO();
        }
        if (JSON.stringify(prev) === JSON.stringify(view)) {
          writeAddress();
          return NO();
        }
        S.decoysChanged = !!prev && prev.decoys !== view.decoys;
        if (prev && prev.t !== view.t) {
          S.tChanged = true;
          setThreshold(view.t);
        }
        hideNotice();
        var api = gridApi(GRIDS.top);
        if (api) {
          api.purgeInfiniteCache();
          api.ensureIndexVisible(0, "top");
        }
        var count = document.getElementById("ib-top-count");
        if (count) {
          count.classList.add("ib-stale");
        }
        writeAddress();
        return NO();
      },

      // The page's first selection.
      sel: function (sel, view, page, built, prec) {
        var fresh = init(page, view, built);
        if (fresh || !S.started) {
          S.sel = sel || {};
          S.prec0 = prec || null;
          S.addrSel = addressSelects(built);
          clearTimeout(S.startTimer);
          // Whatever the first table does, start after a moment.
          S.startTimer = setTimeout(start, 1500);
          var api = gridApi(GRIDS.top);
          var first = api && api.getDisplayedRowAtIndex(0);
          if (first && first.data) {
            requestAnimationFrame(start);
          }
        }
        highlightAll(true);
        writeAddress();
        return NO();
      },

      // The first block of a new view: keep the selection when its row is in the
      // block, else select the first row (or nothing in an empty table).
      first: function (first) {
        var count = document.getElementById("ib-top-count");
        if (count) {
          count.classList.remove("ib-stale");
        }
        if (!first || !S.view) {
          return NO();
        }
        var decoys = S.decoysChanged;
        var tChanged = S.tChanged;
        S.decoysChanged = false;
        S.tChanged = false;
        if (!first.row) {
          clearSelection();
          return NO();
        }
        var key = selKey("top");
        if (key !== null && (first.keys || []).indexOf(key) >= 0) {
          var unit = S.view.unit;
          if (decoys && unit !== "precursor") {
            requestChildren(unit === "protein_group" ? "group" : "peptide", S.sel || {}, null);
          } else if (tChanged) {
            // The marks of the preview follow the threshold.
            wantPreview(S.sel.run, S.sel.cid, null, 0);
          }
          highlight("top", false);
          checkInView();
          return NO();
        }
        pick("top", first.row, { force: true });
        return NO();
      },

      // A children answer: kept, and shown when it answers the newest request.
      children: function (p) {
        if (!p || !S.view) {
          return NO();
        }
        cachePayload(p);
        if (p.kind === "more") {
          applyMore(p.more, p.n);
          return NO();
        }
        if (p.n !== S.nonce || S.appliedN === p.n) {
          return NO();
        }
        applyChildren(p);
        return NO();
      },

      // Answers asked ahead: kept; one may answer the request the page waits for.
      prefetched: function (data) {
        if (!data || !S.view) {
          return NO();
        }
        (data.payloads || []).forEach(cachePayload);
        if (S.need && waitingForChildren() && S.appliedN !== S.need.n) {
          var hit = resolveFromCache(S.need);
          if (hit) {
            applyChildren(hit);
          }
        }
        return NO();
      },

      // A preview card from the server: kept, and inserted when it is the wanted one.
      preview: function (data) {
        if (!data || !data.card) {
          return NO();
        }
        var key = pvKey(data.run, data.cid, data.t);
        PV.cards.set(key, data);
        var wanted = PV.want && PV.want.key === key;
        if (wanted && data.ok === false && S.addrSel && S.view && S.view.unit === "precursor") {
          fallbackToFirst(
            "The precursor in the link (candidate " +
              data.cid +
              (data.run ? ", run " + data.run : "") +
              ") is not in this result set. The first row is selected."
          );
          return NO();
        }
        if (wanted) {
          scheduleInsert();
        }
        return NO();
      },

      // The position of the selected row in the first table (or none within reach).
      located: function (data) {
        if (!data || data.key !== selKey("top")) {
          return NO();
        }
        var api = gridApi(GRIDS.top);
        if (!api) {
          return NO();
        }
        if (data.index === null || data.index === undefined) {
          showNotice(
            "The selected row is not a row of this table with these filters; the panels below still show it."
          );
          return NO();
        }
        api.ensureIndexVisible(data.index, "middle");
        return NO();
      },

      // Another starting level is another layout: a navigation, so the router builds it.
      // The filters stay; the q column (each level has its own) and the selection go.
      level: function (unit, view) {
        if (!view || !unit || unit === view.unit) {
          return NO();
        }
        var next = Object.assign({}, view, { unit: unit, q: "" });
        var el = root();
        if (el && el.dataset.experiment === "1" && unit !== "precursor") {
          // The grouped tables of an experiment are experiment-wide: no run, no quant.
          next.run = "";
          next.quant = "";
        }
        go(window.location.pathname + addressOf(next, null, { col: "score", desc: true }));
        return NO();
      },

      // A chip removes its filter; Reset removes every filter and the sort.
      clear: function () {
        var out = CONTROLS.map(function () {
          return NO();
        });
        var trig = triggered();
        if (!trig || !trig.value) {
          return out;
        }
        var id = trig.prop_id.slice(0, trig.prop_id.lastIndexOf("."));
        var unit = (S.view && S.view.unit) || "protein_group";
        if (id === "ib-reset") {
          var api = gridApi(GRIDS.top);
          if (api) {
            api.applyColumnState({ state: [{ colId: "score", sort: "desc" }], defaultState: { sort: null } });
          }
          return CONTROLS.map(function (k) {
            return k === "q" ? defaultQ({ unit: unit, run: "" }) : CONTROL_DEFAULTS[k];
          });
        }
        var key = null;
        try {
          key = JSON.parse(id).key;
        } catch (e) {
          return out;
        }
        var i = CONTROLS.indexOf(key);
        if (i >= 0) {
          var after = Object.assign({}, S.view || { unit: unit });
          after[key] = key === "decoys" ? false : "";
          out[i] = key === "q" ? defaultQ(after) : CONTROL_DEFAULTS[key];
        }
        if (key === "run" && S.view && !S.view.q) {
          // Without a run the default q column changes back.
          out[CONTROLS.indexOf("q")] = defaultQ({ unit: unit, run: "" });
        }
        return out;
      },

      // A Columns menu shows and hides columns of its grid.
      columns: function (panel, value) {
        var api = gridApi(GRIDS[panel]);
        if (!api) {
          return NO();
        }
        var want = {};
        (value || []).forEach(function (c) {
          want[c] = true;
        });
        var show = [];
        var hide = [];
        api.getColumns().forEach(function (col) {
          var id = col.getColId();
          var def = col.getColDef() || {};
          if (FIXED[id] || def.lockVisible) {
            return;
          }
          (want[id] ? show : hide).push(id);
        });
        api.setColumnsVisible(show, true);
        api.setColumnsVisible(hide, false);
        return NO();
      },

      // The first table's default columns (then fitted to the grid's width again).
      defaultColumns: function (n, defaults) {
        if (!n) {
          return NO();
        }
        setTimeout(function () {
          applyNarrow("top", gridApi(GRIDS.top), true);
        }, 60);
        return defaults || [];
      },
    },
  });

  // For the tests (tests/js): the pure functions of this file.
  window.mvbInternals = {
    addressOf: addressOf,
    norm: norm,
    staticInfo: staticInfo,
    islandize: islandize,
    deferGraphs: deferGraphs,
    collapseTree: collapseTree,
    pickBy: pickBy,
    pickPrecursor: pickPrecursor,
    resolveFromCache: resolveFromCache,
    passCount: passCount,
    partKey: partKey,
    LRU: LRU,
  };
})();
