/* mumdia-viewer: the precursor page (ids pd-*) and its preview on the identification
 * page (ids pv-*).
 *
 * Links the XIC, the spectrum, the sequence diagram, the ion table and the fragment list
 * around one scan:
 *  - the shown scan (XIC click, scan slider, prev / apex / next buttons, arrow keys),
 *    with the spectrum and the tables dimmed while the server sends the new scan;
 *  - fragment highlighting (hover a trace, a peak, a diagram mark, an ion table cell or
 *    a list row), kept when the server rebuilds a panel for a new scan;
 *  - fragment visibility (legend click, legend double click, a click on a fragment);
 *  - the XIC's two views (the peak, the whole RT window) with the scan slider lined up
 *    with the shown range.
 * Fragment traces carry meta = {frag: k, op: base opacity}; k is the same fragment in
 * every panel and the data-frag attribute of its elements. Highlighting works inside
 * one ".pd-linked" group (the page, or the preview card); visibility on the page only.
 */
(function () {
  "use strict";

  const XIC_WRAP = "pd-xic-wrap";
  const XIC_ID = { name: "pd-xic", type: "fig" };
  const DIM = 0.14;
  const GROUP = ".pd-linked";
  const NAV_KEY = "mv-nav";
  const PENDING_MAX = 6000;

  // The page state the clientside callbacks share (reset by `init` on every page).
  const cur = {
    key: null,
    row: null,
    index: null,
    rt: null,
    hidden: [],
    timer: null,
    views: null,
    view: null,
    grid: null,
    nav: null,
    pendingTimer: null,
    scrubTimer: null,
    scrubSet: [],
  };
  let legendTimer = null;

  function NO() {
    return window.dash_clientside.no_update;
  }

  function setProps(id, props) {
    const dc = window.dash_clientside;
    if (dc && typeof dc.set_props === "function") {
      dc.set_props(id, props);
    }
  }

  function plotOf(wrap) {
    const w = document.getElementById(wrap);
    return w ? w.querySelector(".js-plotly-plot") : null;
  }

  function fragOf(trace) {
    const m = trace && trace.meta;
    return m && typeof m === "object" && typeof m.frag === "number" ? m.frag : null;
  }

  function keyOf(k) {
    return k ? String(k.run) + "|" + String(k.cid) : null;
  }

  function triggered() {
    const ctx = window.dash_clientside.callback_context;
    return ctx && ctx.triggered && ctx.triggered.length ? ctx.triggered[0].prop_id : "";
  }

  function isSet(k) {
    return k !== null && k !== undefined;
  }

  // ------------------------------------------------------------------ figures

  // The shown scan over the XIC: an overlay (#pd-scanmark) one scan step wide, placed
  // from the plot's x axis. Moving it does not redraw the plot (a Plotly shape would take
  // about 0.1 s, and the scan's request would wait for it). Placed again after every
  // redraw of the plot (zoom, resize, theme, axis switch) and on every new scan.
  function placeBand() {
    const wrap = document.getElementById(XIC_WRAP);
    const mark = document.getElementById("pd-scanmark");
    const gd = plotOf(XIC_WRAP);
    if (!wrap || !mark || !gd || !gd._fullLayout || !gd._fullLayout.xaxis) {
      return;
    }
    const meta = (gd.layout && gd.layout.meta) || {};
    const fl = gd._fullLayout;
    const size = fl._size;
    const range = fl.xaxis.range;
    // Only the page that set `cur`: a new page draws before `init` has run.
    const ours = !meta.key || meta.key === cur.key;
    const rt = cur.rt;
    if (ours) {
      mark.setAttribute("data-rt", rt === null || rt === undefined ? "" : String(rt));
    }
    if (!ours || rt === null || rt === undefined || !range || range[1] === range[0]) {
      mark.style.display = "none";
      return;
    }
    const span = range[1] - range[0];
    const x = size.l + ((rt - range[0]) / span) * size.w;
    if (x < size.l - 1 || x > size.l + size.w + 1) {
      mark.style.display = "none";
      return;
    }
    const step = typeof meta.step === "number" && meta.step > 0 ? meta.step : 1;
    const width = Math.max(3, (step / span) * size.w);
    const g = gd.getBoundingClientRect();
    const w = wrap.getBoundingClientRect();
    mark.style.display = "block";
    mark.style.left = (g.left - w.left + x).toFixed(1) + "px";
    mark.style.top = (g.top - w.top + size.t).toFixed(1) + "px";
    mark.style.height = size.h.toFixed(1) + "px";
    mark.style.setProperty("--pd-band-w", width.toFixed(1) + "px");
  }

  // A figure with the fragments in `hidden` hidden ("legendonly" keeps the XIC legend).
  function withHidden(fig, hidden, legendonly) {
    if (!fig || !fig.data) {
      return NO();
    }
    const off = new Set(hidden || []);
    let changed = false;
    const data = fig.data.map(function (t) {
      const k = fragOf(t);
      if (k === null) {
        return t;
      }
      const want = off.has(k) ? (legendonly ? "legendonly" : false) : true;
      const now = t.visible === undefined ? true : t.visible;
      if (now === want) {
        return t;
      }
      changed = true;
      return Object.assign({}, t, { visible: want });
    });
    return changed ? Object.assign({}, fig, { data: data }) : NO();
  }

  // The intensity axes of the XIC for its view: the view's ranges (layout.meta.y and
  // .y2, from the traces inside the view) on a linear axis, autorange on a log axis.
  function yAxes(layout, view) {
    const meta = layout.meta || {};
    [
      ["yaxis", meta.y],
      ["yaxis2", meta.y2],
    ].forEach(function (pair) {
      const name = pair[0];
      const ranges = pair[1];
      const ax = Object.assign({}, layout[name] || {});
      const r = ranges && ranges[view];
      if (ax.type === "log" || !r) {
        if (ax.type === "log") {
          ax.autorange = true;
          delete ax.range;
        }
      } else {
        ax.range = r.slice();
        ax.autorange = false;
      }
      layout[name] = ax;
    });
    return layout;
  }

  // The edge notes of the window bounds: shown while the bound lies outside `range`.
  function edgeNotes(annotations, edges, range) {
    if (!annotations || !edges || !range) {
      return annotations;
    }
    const byName = {};
    edges.forEach(function (e) {
      byName[e.name] = e;
    });
    return annotations.map(function (a) {
      const e = a && byName[a.name];
      if (!e) {
        return a;
      }
      const show = e.side === "always" || (e.side === "lo" ? e.x < range[0] : e.x > range[1]);
      return Object.assign({}, a, { visible: show });
    });
  }

  // ------------------------------------------------------------------ highlight

  function restylePlot(gd, k) {
    if (!gd || !gd.data || !window.Plotly) {
      return;
    }
    const idx = [];
    const op = [];
    gd.data.forEach(function (t, i) {
      const f = fragOf(t);
      if (f === null) {
        return;
      }
      const base = typeof t.meta.op === "number" ? t.meta.op : 1;
      idx.push(i);
      op.push(!isSet(k) ? base : f === k ? 1 : base * DIM);
    });
    gd.__pdData = gd.data;
    if (idx.length) {
      window.Plotly.restyle(gd, { opacity: op }, idx);
    }
  }

  function markElements(group, k) {
    group.querySelectorAll("[data-frag]").forEach(function (el) {
      const f = Number(el.getAttribute("data-frag"));
      el.classList.toggle("pd-hl", isSet(k) && f === k);
      el.classList.toggle("pd-dim", isSet(k) && f !== k);
    });
  }

  // Highlight fragment k (or none) in every plot and fragment element of one group.
  // `src` is the plot the hover came from (null for an element).
  function highlight(group, k, src) {
    if (!group || group.__pdHover === k) {
      return;
    }
    group.__pdHover = k;
    group.__pdSrc = src || null;
    group.querySelectorAll(".js-plotly-plot").forEach(function (gd) {
      restylePlot(gd, k);
    });
    markElements(group, k);
  }

  // A plot drew a new figure (a new scan, a hidden fragment, a theme): give it the
  // group's highlight. A highlight that came from that very plot ends, since the plot
  // lost its hover state.
  function afterPlot(gd) {
    if (!gd || !gd.data || gd.data === gd.__pdData) {
      return;
    }
    const group = gd.closest(GROUP);
    if (!group) {
      return;
    }
    if (group.__pdSrc === gd && isSet(group.__pdHover)) {
      highlight(group, null);
      return;
    }
    if (isSet(group.__pdHover)) {
      restylePlot(gd, group.__pdHover);
    } else {
      gd.__pdData = gd.data;
    }
  }

  function onHover(gd, ev) {
    const p = ev && ev.points && ev.points[0];
    highlight(gd.closest(GROUP), p ? fragOf(p.data) : null, gd);
  }

  function onUnhover(gd) {
    highlight(gd.closest(GROUP), null);
  }

  // ------------------------------------------------------------------ visibility

  function markHidden() {
    document.querySelectorAll(".pd-page [data-frag]").forEach(function (el) {
      const f = Number(el.getAttribute("data-frag"));
      el.classList.toggle("pd-off", cur.hidden.indexOf(f) >= 0);
    });
  }

  function setHidden(list) {
    cur.hidden = list.slice().sort(function (a, b) {
      return a - b;
    });
    setProps("pd-hidden", { data: cur.hidden });
    markHidden();
  }

  function toggle(k) {
    const off = new Set(cur.hidden);
    if (off.has(k)) {
      off.delete(k);
    } else {
      off.add(k);
    }
    setHidden(Array.from(off));
  }

  function allFragments() {
    const gd = plotOf(XIC_WRAP);
    const out = [];
    ((gd && gd.data) || []).forEach(function (t) {
      const k = fragOf(t);
      if (k !== null && out.indexOf(k) < 0) {
        out.push(k);
      }
    });
    return out;
  }

  function isolate(k) {
    const others = allFragments().filter(function (i) {
      return i !== k;
    });
    const alone =
      cur.hidden.indexOf(k) < 0 &&
      others.every(function (i) {
        return cur.hidden.indexOf(i) >= 0;
      });
    setHidden(alone ? [] : others);
  }

  function onLegendClick(ev) {
    const k = fragOf(ev && ev.data && ev.data[ev.curveNumber]);
    if (k === null) {
      return true;
    }
    if (legendTimer) {
      clearTimeout(legendTimer);
    }
    // Wait for a possible double click (which isolates the fragment instead).
    legendTimer = setTimeout(function () {
      legendTimer = null;
      toggle(k);
    }, 280);
    return false;
  }

  function onLegendDoubleClick(ev) {
    const k = fragOf(ev && ev.data && ev.data[ev.curveNumber]);
    if (k === null) {
      return true;
    }
    if (legendTimer) {
      clearTimeout(legendTimer);
      legendTimer = null;
    }
    isolate(k);
    return false;
  }

  // ------------------------------------------------------------------ pending scan

  // The spectrum and the tables show the previous scan until the server answers: dim
  // them meanwhile (detail.css, .pd-pending), at most PENDING_MAX ms.
  function setPending(on) {
    const page = document.querySelector(".pd-page");
    if (cur.pendingTimer) {
      clearTimeout(cur.pendingTimer);
      cur.pendingTimer = null;
    }
    if (!page) {
      return;
    }
    page.classList.toggle("pd-pending", on);
    if (on) {
      cur.pendingTimer = setTimeout(function () {
        setPending(false);
      }, PENDING_MAX);
    }
  }

  // ------------------------------------------------------------------ navigation

  // The last two addresses of this tab, so that "Identifications" can go back to the
  // identification page as the user left it (search, filters, selection).
  function recordNav() {
    try {
      const here = window.location.pathname + window.location.search;
      const nav = JSON.parse(window.sessionStorage.getItem(NAV_KEY) || "{}") || {};
      if (nav.cur !== here) {
        nav.prev = nav.cur || null;
        nav.cur = here;
        window.sessionStorage.setItem(NAV_KEY, JSON.stringify(nav));
      }
    } catch (e) {
      /* storage blocked: the link's own address is used */
    }
  }

  function cameFromIdentifications() {
    try {
      const nav = JSON.parse(window.sessionStorage.getItem(NAV_KEY) || "{}") || {};
      return Boolean(nav.prev) && /\/identifications(\?|$)/.test(String(nav.prev));
    } catch (e) {
      return false;
    }
  }

  // ------------------------------------------------------------------ tooltip

  // One tooltip for the elements with data-tip (diagram marks, ion table cells, table
  // headers): rebuilt on every scan step, a Mantine tooltip on each would make the step
  // slower to draw. Styled like Mantine's (detail.css, .pd-tip).
  const TIP_DELAY = 300;
  let tipEl = null;
  let tipTimer = null;
  let tipFor = null;

  function tipNode() {
    if (!tipEl) {
      tipEl = document.createElement("div");
      tipEl.className = "pd-tip";
      tipEl.setAttribute("role", "tooltip");
      document.body.appendChild(tipEl);
    }
    return tipEl;
  }

  function showTip(el) {
    const text = el.getAttribute("data-tip");
    if (!text || !el.isConnected) {
      return;
    }
    const t = tipNode();
    t.textContent = text;
    t.style.display = "block";
    const r = el.getBoundingClientRect();
    const tw = t.offsetWidth;
    const th = t.offsetHeight;
    let x;
    let y;
    if (el.getAttribute("data-tip-side") === "left") {
      x = r.left - tw - 8;
      y = r.top + r.height / 2 - th / 2;
      if (x < 4) {
        x = r.right + 8;
      }
    } else {
      x = r.left + r.width / 2 - tw / 2;
      y = r.top - th - 8;
      if (y < 4) {
        y = r.bottom + 8;
      }
    }
    t.style.left = Math.max(4, Math.min(window.innerWidth - tw - 4, x)).toFixed(0) + "px";
    t.style.top = Math.max(4, y).toFixed(0) + "px";
  }

  function hideTip() {
    if (tipTimer) {
      clearTimeout(tipTimer);
      tipTimer = null;
    }
    tipFor = null;
    if (tipEl) {
      tipEl.style.display = "none";
    }
  }

  function tipTarget(node) {
    const el = node && node.closest ? node.closest("[data-tip]") : null;
    return el && el.closest(GROUP) ? el : null;
  }

  // ------------------------------------------------------------------ events

  let bindQueued = false;

  function bind() {
    bindQueued = false;
    recordNav();
    document.querySelectorAll(GROUP + " .js-plotly-plot").forEach(function (gd) {
      if (typeof gd.on !== "function" || gd.__pdBound) {
        return;
      }
      gd.__pdBound = true;
      gd.on("plotly_hover", function (ev) {
        onHover(gd, ev);
      });
      gd.on("plotly_unhover", function () {
        onUnhover(gd);
      });
      gd.on("plotly_afterplot", function () {
        afterPlot(gd);
      });
      if (gd.closest("#" + XIC_WRAP)) {
        gd.on("plotly_legendclick", onLegendClick);
        gd.on("plotly_legenddoubleclick", onLegendDoubleClick);
        gd.on("plotly_afterplot", placeBand);
        placeBand();
      }
    });
    // Elements rebuilt by the server (a new scan) keep the highlight and the hidden
    // fragments; the tooltip of a replaced element closes.
    document.querySelectorAll(GROUP).forEach(function (group) {
      if (isSet(group.__pdHover)) {
        markElements(group, group.__pdHover);
      }
    });
    if (cur.hidden.length) {
      markHidden();
    }
    if (tipFor && !tipFor.isConnected) {
      hideTip();
    }
  }

  function queueBind() {
    if (!bindQueued) {
      bindQueued = true;
      window.requestAnimationFrame(bind);
    }
  }

  function fragElement(target) {
    const el = target && target.closest ? target.closest("[data-frag]") : null;
    return el && el.closest(GROUP) ? el : null;
  }

  function start() {
    new MutationObserver(queueBind).observe(document.body, { childList: true, subtree: true });
    window.addEventListener("popstate", recordNav);
    document.addEventListener("mouseover", function (ev) {
      const el = fragElement(ev.target);
      if (el) {
        highlight(el.closest(GROUP), Number(el.getAttribute("data-frag")), null);
      }
      const t = tipTarget(ev.target);
      if (t !== tipFor) {
        hideTip();
        if (t) {
          tipFor = t;
          tipTimer = setTimeout(function () {
            tipTimer = null;
            if (tipFor === t) {
              showTip(t);
            }
          }, TIP_DELAY);
        }
      }
    });
    document.addEventListener("mouseout", function (ev) {
      const el = fragElement(ev.target);
      if (el && !el.contains(ev.relatedTarget)) {
        const next = fragElement(ev.relatedTarget);
        if (!next) {
          highlight(el.closest(GROUP), null);
        }
      }
      if (tipFor && !tipFor.contains(ev.relatedTarget)) {
        hideTip();
      }
    });
    window.addEventListener("scroll", hideTip, true);
    document.addEventListener("click", function (ev) {
      const el = fragElement(ev.target);
      if (el && el.closest(".pd-page")) {
        toggle(Number(el.getAttribute("data-frag")));
      }
    });
    // The scan buttons (and the arrow keys, which click them) step at their click.
    document.addEventListener(
      "click",
      function (ev) {
        const b = ev.target && ev.target.closest ? ev.target.closest("#scan-prev, #scan-next, #scan-apex") : null;
        if (!b || b.disabled || !cur.grid) {
          return;
        }
        ev.preventDefault();
        ev.stopPropagation();
        stepScan(b.id === "scan-prev" ? "prev" : b.id === "scan-next" ? "next" : "apex");
      },
      true
    );
    // "Identifications" goes back in the history when the previous page was the
    // identification page (before the link's own handler runs).
    document.addEventListener(
      "click",
      function (ev) {
        const a = ev.target && ev.target.closest ? ev.target.closest("a.pd-back") : null;
        if (!a || ev.button !== 0 || ev.metaKey || ev.ctrlKey || ev.shiftKey) {
          return;
        }
        if (cameFromIdentifications() && window.history.length > 1) {
          ev.preventDefault();
          ev.stopPropagation();
          window.history.back();
        }
      },
      true
    );
  }

  if (document.body) {
    start();
  } else {
    document.addEventListener("DOMContentLoaded", start);
  }

  // The thumb label of the scan slider: the RT and scan_index of the grid point.
  window.dashMantineFunctions = Object.assign({}, window.dashMantineFunctions, {
    pdScanLabel: function (value, options) {
      const o = options || {};
      const i = Math.round(value);
      const rt = o.rt && o.rt[i] !== undefined ? o.rt[i].toFixed(2) + " s" : String(i);
      const scan = o.scan && o.scan[i] !== undefined ? " · scan " + o.scan[i] : "";
      return rt + scan;
    },
  });

  function nearest(grid, x) {
    const rt32 = grid.rt32 || [];
    let best = -1;
    let dist = Infinity;
    for (let i = 0; i < rt32.length; i += 1) {
      if (rt32[i] === x) {
        return i;
      }
      const dd = Math.abs(rt32[i] - x);
      if (dd < dist) {
        dist = dd;
        best = i;
      }
    }
    return best >= 0 ? best : null;
  }

  // The XIC leaves the peak view for the whole window when the shown scan leaves it.
  function followScan(index) {
    if (index === null || index === undefined || cur.view !== "peak" || !cur.views) {
      return;
    }
    const peak = cur.views.peak;
    const s = peak && peak.slider;
    if (s && (index < s.min || index > s.max)) {
      cur.view = "window";
      setProps("pd-xic-view", { value: "window" });
    }
  }

  // A grid point (or a row outside the grid) as the shown scan's selection.
  function pick(grid, index, row, rt) {
    if (index !== null && index !== undefined) {
      return { index: index, row: grid.rows[index], rt: grid.rt32[index] };
    }
    return row === null || row === undefined ? null : { index: null, row: row, rt: rt };
  }

  // The scan the previous, next or apex button shows from the shown one: the neighbour
  // in the grid, beyond the grid the same-window neighbour the server gave (nav).
  function target(kind, grid, nav) {
    if (!grid || !grid.rows) {
      return null;
    }
    const n = grid.rows.length;
    if (kind === "apex") {
      if (grid.apex !== null && grid.apex !== undefined) {
        return pick(grid, grid.apex, null, null);
      }
      return nav ? pick(grid, null, nav.apex_row, nav.apex_rt) : null;
    }
    const step = kind === "prev" ? -1 : 1;
    if (cur.index !== null && cur.index + step >= 0 && cur.index + step < n) {
      return pick(grid, cur.index + step, null, null);
    }
    const near = nav || {};
    const r = step < 0 ? near.prev_row : near.next_row;
    if (r === null || r === undefined || near.row !== cur.row) {
      return null;
    }
    const at = grid.rows.indexOf(r);
    return at >= 0 ? pick(grid, at, null, null) : pick(grid, null, r, step < 0 ? near.prev_rt : near.next_rt);
  }

  // Make `sel` the shown scan: the band moves, the XIC follows, the spectrum is pending.
  function commit(sel) {
    cur.row = sel.row;
    cur.index = sel.index;
    cur.rt = sel.rt;
    placeBand();
    followScan(sel.index);
    setPending(true);
    if (cur.timer) {
      clearTimeout(cur.timer);
      cur.timer = null;
    }
    return { row: sel.row, index: sel.index, rt: sel.rt };
  }

  // A scan button (or its arrow key, which clicks it): the step is sent at once, without
  // the round through the button's n_clicks, which costs about 0.1 s on this page.
  function stepScan(kind) {
    const sel = target(kind, cur.grid, cur.nav);
    if (!sel) {
      return;
    }
    const data = commit(sel);
    setProps("pd-scan", { data: data });
    // The slider follows a moment later: updating it first would hold the scan's
    // request back by about 50 ms.
    if (cur.scrubTimer) {
      clearTimeout(cur.scrubTimer);
    }
    cur.scrubTimer = setTimeout(function () {
      cur.scrubTimer = null;
      if (cur.index !== null && cur.index !== undefined) {
        // The slider callback must not take this value for a drag (a key pressed
        // meanwhile has moved the scan on).
        cur.scrubSet.push(cur.index);
        setProps("pd-scrub", { value: cur.index });
      }
    }, 60);
  }

  // ------------------------------------------------------------------ callbacks

  window.dash_clientside = Object.assign({}, window.dash_clientside, {
    mvd: {
      // A new precursor page: forget the previous page's state.
      init: function (key, grid, scan, views, nav) {
        if (cur.timer) {
          clearTimeout(cur.timer);
        }
        if (cur.scrubTimer) {
          clearTimeout(cur.scrubTimer);
          cur.scrubTimer = null;
        }
        cur.grid = grid || null;
        cur.nav = nav || null;
        cur.scrubSet = [];
        setPending(false);
        cur.key = keyOf(key);
        cur.row = scan ? scan.row : null;
        cur.index = scan ? scan.index : null;
        cur.rt = scan ? scan.rt : null;
        cur.hidden = [];
        cur.timer = null;
        cur.views = views || null;
        cur.view = views ? views.view : null;
        placeBand();
        return Date.now();
      },

      // The shown scan from an XIC click or the slider (the scan buttons and the arrow
      // keys are handled at their click, see stepScan). The XIC band moves at once; a
      // slider drag sends the scan to the server when it rests.
      scan: function (click, scrub, nPrev, nApex, nNext, grid, scan, nav) {
        const trig = triggered();
        const none = [NO(), NO()];
        if (!grid || !grid.rows) {
          return none;
        }
        if (cur.row === null && scan) {
          cur.row = scan.row;
          cur.index = scan.index;
          cur.rt = scan.rt;
        }
        const n = grid.rows.length;
        let sel = null;
        const fromSlider = trig.indexOf("pd-scrub") >= 0;
        if (trig.indexOf("clickData") >= 0) {
          if (!click || !click.points || !click.points.length) {
            return none;
          }
          sel = pick(grid, nearest(grid, click.points[0].x), null, null);
          // Clear the click so that a second click on the same point is seen.
          setTimeout(function () {
            setProps(XIC_ID, { clickData: null });
          }, 0);
        } else if (fromSlider) {
          if (scrub === null || scrub === undefined || !n) {
            return none;
          }
          const index = Math.max(0, Math.min(n - 1, Math.round(scrub)));
          const given = cur.scrubSet.indexOf(index);
          if (given >= 0) {
            // A value stepScan gave the slider, not a drag.
            cur.scrubSet.splice(0, given + 1);
            return none;
          }
          if (index === cur.index) {
            return none;
          }
          sel = pick(grid, index, null, null);
        } else {
          // A button's n_clicks (when its click was not handled by stepScan).
          let kind = null;
          ["prev", "next", "apex"].forEach(function (k) {
            if (trig.indexOf("scan-" + k) >= 0) {
              kind = k;
            }
          });
          sel = kind ? target(kind, grid, nav) : null;
        }
        if (!sel) {
          return none;
        }
        const data = commit(sel);
        if (fromSlider) {
          cur.timer = setTimeout(function () {
            cur.timer = null;
            setProps("pd-scan", { data: data });
          }, 160);
          return [NO(), NO()];
        }
        return [data, sel.index !== null ? sel.index : NO()];
      },

      // The server answered with the shown scan (pd-nav is its last output).
      settled: function (nav) {
        cur.nav = nav || null;
        setPending(false);
        return Date.now();
      },

      // The fragment card shows the list or the ion table.
      fragView: function (value) {
        const ladder = value === "ladder";
        return [ladder ? { display: "none" } : {}, ladder ? {} : { display: "none" }];
      },

      // The server's mirror of the new scan, with the zoom of the shown one. Dash writes
      // a user's zoom into the graph's figure, so Plotly's uirevision cannot keep it when
      // the server sends a figure; the server's range is the candidate's (the same for
      // every scan), so a range that differs from it is the user's.
      mirror: function (next, current) {
        if (!next || !next.layout) {
          return NO();
        }
        const layout = Object.assign({}, next.layout);
        const cl = (current && current.layout) || {};
        ["xaxis", "yaxis"].forEach(function (name) {
          const ca = cl[name] || {};
          const na = Object.assign({}, layout[name] || {});
          if (ca.autorange === true) {
            na.autorange = true;
            delete na.range;
          } else if (
            Array.isArray(ca.range) &&
            Array.isArray(na.range) &&
            (ca.range[0] !== na.range[0] || ca.range[1] !== na.range[1])
          ) {
            na.range = ca.range.slice();
            na.autorange = false;
          }
          layout[name] = na;
        });
        return Object.assign({}, next, { layout: layout });
      },

      // Fragment visibility in both plots.
      hidden: function (hidden, xic, mirror) {
        return [withHidden(xic, hidden, true), withHidden(mirror, hidden, false)];
      },

      // Linear or log intensity axes of the XIC (both rows); a linear axis takes the
      // range of the shown view again.
      axis: function (value, fig) {
        if (!fig || !fig.layout) {
          return NO();
        }
        const log = value === "log";
        const layout = Object.assign({}, fig.layout);
        ["yaxis", "yaxis2"].forEach(function (name) {
          const ax = Object.assign({}, layout[name] || {});
          ax.type = log ? "log" : "linear";
          ax.autorange = true;
          delete ax.range;
          layout[name] = ax;
        });
        const view = (layout.meta && layout.meta.view) || "window";
        return Object.assign({}, fig, { layout: yAxes(layout, view) });
      },

      // The XIC's peak view or its whole window, with the scan slider over the grid
      // scans in view, lined up with them, and the window's edge notes.
      view: function (value, fig, views) {
        const none = [NO(), NO(), NO(), NO(), NO()];
        const v = views && (views[value] || views.window);
        if (!fig || !fig.layout || !v || !v.range) {
          return none;
        }
        cur.view = views[value] ? value : "window";
        const layout = Object.assign({}, fig.layout);
        ["xaxis", "xaxis2"].forEach(function (name) {
          const ax = Object.assign({}, layout[name] || {});
          ax.range = v.range.slice();
          ax.autorange = false;
          layout[name] = ax;
        });
        const meta = Object.assign({}, layout.meta || {}, { view: cur.view });
        layout.meta = meta;
        layout.annotations = edgeNotes(layout.annotations, meta.edges, v.range);
        const out = Object.assign({}, fig, { layout: yAxes(layout, cur.view) });
        const s = v.slider;
        if (!s) {
          return [out, NO(), NO(), NO(), NO()];
        }
        return [out, s.min, s.max, s.marks || [], s.style || {}];
      },

      // A click on a competition dot opens that row (through its link, so the page does
      // not reload).
      open: function (click) {
        const p = click && click.points && click.points[0];
        const target = p && typeof p.customdata === "string" ? p.customdata : null;
        if (!target) {
          return NO();
        }
        const links = document.querySelectorAll("a[href]");
        for (let i = 0; i < links.length; i += 1) {
          if (links[i].getAttribute("href") === target) {
            links[i].click();
            return NO();
          }
        }
        return target;
      },
    },
  });
})();
