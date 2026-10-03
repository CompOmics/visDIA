/* mumdia-viewer: the spectrum browser (sp-*). Client-side callbacks (mvs), the cell
   renderers of the candidate list, its AG Grid theme, the step buttons and the m/z labels
   that follow the zoom. */

(function () {
  "use strict";

  function NO() {
    return window.dash_clientside.no_update;
  }

  function h() {
    return window.React.createElement.apply(window.React, arguments);
  }

  // In-app navigation, as dcc.Link does it: the router builds the page, no reload.
  function go(url) {
    if (!url) {
      return;
    }
    window.history.pushState({}, "", url);
    window.dispatchEvent(new CustomEvent("_dashprivate_pushstate"));
    window.scrollTo(0, 0);
  }

  function plainClick(ev) {
    return ev.button === 0 && !ev.metaKey && !ev.ctrlKey && !ev.shiftKey && !ev.altKey;
  }

  // The precursor page of a candidate (ui.state.href with run and cid).
  function precursorUrl(ctx, cid) {
    var q = new URLSearchParams();
    if (ctx.run) {
      q.set("run", ctx.run);
    }
    q.set("cid", String(cid));
    return (ctx.base || "/") + "precursor?" + q.toString();
  }

  // ------------------------------------------------------------------ step buttons

  // The step buttons (and the arrow keys, which click them) set sp-step. The listener
  // runs in the capture phase of the window, before any document listener of another
  // page's script, so no other handler can swallow the click.
  window.addEventListener(
    "click",
    function (ev) {
      var b = ev.target && ev.target.closest ? ev.target.closest("#scan-prev, #scan-next") : null;
      if (!b || !b.closest(".sp-page") || b.disabled) {
        return;
      }
      ev.preventDefault();
      ev.stopPropagation();
      var n = b.id === "scan-prev" ? -1 : 1;
      if (window.dash_clientside && window.dash_clientside.set_props) {
        window.dash_clientside.set_props("sp-step", { data: { n: n, ts: Date.now() } });
      }
    },
    true
  );

  // The related-scan buttons of the scan card go to their scan.
  document.addEventListener("click", function (ev) {
    var b = ev.target && ev.target.closest ? ev.target.closest(".sp-page .sp-jump") : null;
    if (!b || !window.dash_clientside || !window.dash_clientside.set_props) {
      return;
    }
    ev.preventDefault();
    window.dash_clientside.set_props("sp-goto", {
      data: {
        level: Number(b.getAttribute("data-level")),
        row: Number(b.getAttribute("data-row")),
        ts: Date.now(),
      },
    });
  });

  // A segmented control keeps the focus on its radio input, where the arrow keys would
  // change its value instead of stepping scans: give the focus back after a change.
  document.addEventListener("change", function (ev) {
    var el = ev.target;
    if (el && el.type === "radio" && el.closest && el.closest(".sp-page")) {
      setTimeout(function () {
        el.blur();
      }, 0);
    }
  });

  // ------------------------------------------------------------------ grid cells

  var cells = (window.dashAgGridComponentFunctions = window.dashAgGridComponentFunctions || {});

  // Matched library fragments in the shown scan: "8/12" with a small bar.
  cells.SpMatched = function (p) {
    var row = p.data || {};
    if (row.matched === undefined || row.matched === null) {
      return h("span", { className: "sp-dim", title: "not computed for this scan" }, "-");
    }
    var n = Number(row.n_lib) || 0;
    var frac = n > 0 ? Math.min(1, Number(row.matched) / n) : 0;
    return h(
      "span",
      {
        className: "sp-match",
        title:
          row.matched + " of " + n + " library fragments match a peak of this scan (viewer match)",
      },
      h(
        "span",
        { className: "sp-match-bar" },
        h("span", {
          className: "sp-match-fill",
          style: { display: "block", width: (100 * frac).toFixed(0) + "%" },
        })
      ),
      row.matched + "/" + n
    );
  };

  // The arrow that opens the candidate's precursor page.
  cells.SpOpen = function (p) {
    if (!p.data) {
      return null;
    }
    var ctx = p.context || {};
    var url = precursorUrl(ctx, p.data.cid);
    var mask = ctx.icon ? 'url("' + ctx.icon + '")' : "none";
    return h(
      "a",
      {
        className: "sp-open",
        href: url,
        title: "Open the precursor page of candidate " + p.data.cid,
        onClick: function (ev) {
          ev.stopPropagation();
          if (!plainClick(ev)) {
            return;
          }
          ev.preventDefault();
          go(url);
        },
      },
      h("span", { className: "sp-open-icon", style: { maskImage: mask, WebkitMaskImage: mask } })
    );
  };

  // ------------------------------------------------------------------ grid theme

  var F = {
    mvsTheme: function (base) {
      return base.withParams({
        fontFamily: "inherit",
        fontSize: 12.5,
        headerFontSize: 12,
        headerFontWeight: 600,
        backgroundColor: "var(--sp-grid-bg)",
        foregroundColor: "var(--sp-grid-fg)",
        textColor: "var(--sp-grid-fg)",
        subtleTextColor: "var(--sp-grid-dim)",
        borderColor: "var(--sp-grid-border)",
        chromeBackgroundColor: "var(--sp-grid-chrome)",
        headerBackgroundColor: "var(--sp-grid-header)",
        headerTextColor: "var(--sp-grid-header-fg)",
        rowHoverColor: "var(--sp-grid-hover)",
        selectedRowBackgroundColor: "var(--sp-grid-selected)",
        accentColor: "var(--sp-grid-accent)",
        wrapperBorder: false,
        wrapperBorderRadius: 0,
        rowBorder: { style: "solid", width: 1, color: "var(--sp-grid-rowline)" },
        columnBorder: false,
        headerRowBorder: { style: "solid", width: 1, color: "var(--sp-grid-border)" },
        spacing: 4,
        cellHorizontalPadding: 7,
        rowHeight: 27,
        headerHeight: 28,
        iconSize: 13,
        headerColumnResizeHandleColor: "transparent",
        browserColorScheme: "inherit",
        tooltipBackgroundColor: "var(--sp-tip-bg)",
        tooltipTextColor: "var(--sp-tip-fg)",
        tooltipBorder: false,
      });
    },
  };
  window.dashAgGridFunctions = Object.assign(window.dashAgGridFunctions || {}, F);

  // The RT slider's thumb label.
  window.dashMantineFunctions = Object.assign({}, window.dashMantineFunctions, {
    spRtLabel: function (value) {
      return Number(value).toFixed(1) + " s";
    },
  });

  // ------------------------------------------------------------------ m/z labels

  var CLIP = 110;

  // The most intense peaks in [lo, hi], at least 1.5 % of the range apart (the rule of
  // data.scans.top_peaks), leaving out the peaks a fragment label already names.
  function topPeaks(xs, rel, n, lo, hi, skip) {
    var idx = [];
    for (var i = 0; i < xs.length; i++) {
      if (xs[i] >= lo && xs[i] <= hi) {
        idx.push(i);
      }
    }
    idx.sort(function (a, b) {
      return rel[b] - rel[a] || a - b;
    });
    var gap = 0.015 * Math.max(hi - lo, 1e-9);
    var chosen = [];
    for (var k = 0; k < idx.length && chosen.length < n; k++) {
      var x = xs[idx[k]];
      var near = false;
      for (var j = 0; j < chosen.length; j++) {
        if (Math.abs(x - xs[chosen[j]]) < gap) {
          near = true;
          break;
        }
      }
      for (var s = 0; !near && s < skip.length; s++) {
        if (Math.abs(x - skip[s]) < 1e-3) {
          near = true;
        }
      }
      if (!near) {
        chosen.push(idx[k]);
      }
    }
    chosen.sort(function (a, b) {
      return xs[a] - xs[b];
    });
    return chosen;
  }

  // Label heights in tiers, as spectra_figures.label_heights places them.
  function heights(xs, ys, span) {
    var width = span * 0.06;
    var tier = 7;
    var placed = [];
    return xs.map(function (x, i) {
      var y = Math.min(ys[i], CLIP) + 3;
      var moved = true;
      while (moved && y < 150) {
        moved = false;
        for (var k = 0; k < placed.length; k++) {
          if (Math.abs(x - placed[k][0]) < width && Math.abs(y - placed[k][1]) < tier) {
            y += tier;
            moved = true;
            break;
          }
        }
      }
      placed.push([x, y]);
      return Math.round(y * 10) / 10;
    });
  }

  function relabel() {
    var gd = document.querySelector("#sp-spec-card .js-plotly-plot");
    if (!gd || !gd.data || !gd.layout || !window.Plotly) {
      return;
    }
    var peaks = -1;
    var labels = -1;
    var skip = [];
    gd.data.forEach(function (t, i) {
      var kind = t.meta && t.meta.kind;
      if (kind === "peaks") {
        peaks = i;
      } else if (kind === "labels") {
        labels = i;
      } else if (kind === "matched" && t.x && t.x.length) {
        skip.push(Number(t.x[0]));
      }
    });
    if (peaks < 0 || labels < 0) {
      return;
    }
    var n = Number((gd.data[labels].meta || {}).n) || 0;
    var range = gd.layout.xaxis && gd.layout.xaxis.range;
    if (!range || range.length !== 2) {
      return;
    }
    var lo = Number(range[0]);
    var hi = Number(range[1]);
    // The decoded trace (gd.data may hold the base64 form of a typed array).
    var tr = (gd._fullData && gd._fullData[peaks]) || gd.data[peaks];
    if (!tr.x || tr.x.length === undefined) {
      return;
    }
    var xs = Array.prototype.map.call(tr.x, Number);
    var cd = tr.customdata || [];
    var rel = xs.map(function (_, i) {
      return cd[i] ? Number(cd[i][1]) : 0;
    });
    var chosen = n > 0 ? topPeaks(xs, rel, n, lo, hi, skip) : [];
    var lx = chosen.map(function (i) {
      return xs[i];
    });
    var ly = heights(
      lx,
      chosen.map(function (i) {
        return rel[i];
      }),
      hi - lo
    );
    var lt = lx.map(function (x) {
      return x.toFixed(2);
    });
    var old = gd.data[labels];
    if (
      old.x &&
      old.x.length === lx.length &&
      lx.every(function (x, i) {
        return Number(old.x[i]) === x && Number(old.y[i]) === ly[i];
      })
    ) {
      return;
    }
    window.Plotly.restyle(gd, { x: [lx], y: [ly], text: [lt] }, [labels]);
  }

  // ------------------------------------------------------------------ callbacks

  window.dash_clientside = Object.assign({}, window.dash_clientside, {
    mvs: {
      // Another run of an experiment: the scan nearest the same RT (and level, window).
      run: function (value, key, scan) {
        if (!key || !value || value === key.run) {
          return NO();
        }
        var q = new URLSearchParams({ run: value });
        if (scan) {
          q.set("rt", Number(scan.rt).toFixed(3));
          q.set("level", String(scan.level));
          if (scan.window !== null && scan.window !== undefined) {
            q.set("window", String(scan.window));
          }
        }
        return key.base + "spectra?" + q.toString();
      },

      // A click on a row selects its candidate (the open arrow navigates instead).
      pick: function (cell, current) {
        // cellClicked carries the row id (getRowId: the candidate_id), not the row.
        if (!cell || cell.colId === "_open" || cell.rowId === undefined || cell.rowId === null) {
          return NO();
        }
        var cid = Number(cell.rowId);
        return !isFinite(cid) || cid === current ? NO() : cid;
      },

      // The address follows the shown scan and the selected candidate (no reload).
      address: function (scan, cid, key) {
        if (!scan || !key) {
          return NO();
        }
        if (window.location.pathname !== key.base + "spectra") {
          return NO();
        }
        var q = new URLSearchParams();
        if (key.run) {
          q.set("run", key.run);
        }
        q.set("scan", String(scan.scan_index));
        if (cid !== null && cid !== undefined) {
          q.set("cid", String(cid));
        }
        var url = window.location.pathname + "?" + q.toString();
        if (url !== window.location.pathname + window.location.search) {
          window.history.replaceState(window.history.state, "", url);
        }
        return NO();
      },

      // The m/z labels name the most intense peaks in view, after every zoom and every
      // new spectrum.
      relabel: function () {
        clearTimeout(relabel.timer);
        relabel.timer = setTimeout(relabel, 80);
        return NO();
      },
    },
  });
})();
