/* mumdia-viewer: the run QC page (qc-*). Client-side callbacks (mvqc), the cell
   renderers of the RT-bin list and its AG Grid theme. */

(function () {
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

  // ------------------------------------------------------------------ grid cells

  var cells = (window.dashAgGridComponentFunctions = window.dashAgGridComponentFunctions || {});

  // The peptidoform as a link to its precursor page (the shared PeptideShaker style).
  cells.QcPeptidoform = function (p) {
    if (!p.data) {
      return null;
    }
    // The precursor page, as ui.qc_cards.precursor_link builds it (state.href).
    var ctx = p.context || {};
    var query = new URLSearchParams();
    if (ctx.run) {
      query.set("run", ctx.run);
    }
    query.set("cid", String(p.data.cid));
    var url = (ctx.base || "/") + "precursor?" + query.toString();
    var text = String(p.value || "");
    var body = window.mvPeptidoformElement ? window.mvPeptidoformElement(text) : text;
    return h(
      "a",
      {
        className: "qc-pep-link",
        href: url,
        title: "Open the precursor page of " + text,
        onClick: function (ev) {
          if (!plainClick(ev)) {
            return;
          }
          ev.preventDefault();
          ev.stopPropagation();
          go(url);
        },
      },
      body
    );
  };

  // ------------------------------------------------------------------ grid theme

  var F = {
    // The identification page's dense grid look, with this page's colour variables.
    mvqcTheme: function (base) {
      return base.withParams({
        fontFamily: "inherit",
        fontSize: 12.5,
        headerFontSize: 12,
        headerFontWeight: 600,
        backgroundColor: "var(--qc-grid-bg)",
        foregroundColor: "var(--qc-grid-fg)",
        textColor: "var(--qc-grid-fg)",
        subtleTextColor: "var(--qc-grid-dim)",
        borderColor: "var(--qc-grid-border)",
        chromeBackgroundColor: "var(--qc-grid-chrome)",
        headerBackgroundColor: "var(--qc-grid-header)",
        headerTextColor: "var(--qc-grid-header-fg)",
        rowHoverColor: "var(--qc-grid-hover)",
        selectedRowBackgroundColor: "var(--qc-grid-selected)",
        accentColor: "var(--qc-grid-accent)",
        wrapperBorder: false,
        wrapperBorderRadius: 0,
        rowBorder: { style: "solid", width: 1, color: "var(--qc-grid-rowline)" },
        columnBorder: false,
        headerRowBorder: { style: "solid", width: 1, color: "var(--qc-grid-border)" },
        spacing: 4,
        cellHorizontalPadding: 8,
        rowHeight: 26,
        headerHeight: 28,
        iconSize: 13,
        headerColumnResizeHandleColor: "transparent",
        browserColorScheme: "inherit",
        tooltipBackgroundColor: "var(--qc-tip-bg)",
        tooltipTextColor: "var(--qc-tip-fg)",
        tooltipBorder: false,
      });
    },
  };
  window.dashAgGridFunctions = Object.assign(window.dashAgGridFunctions || {}, F);

  // ------------------------------------------------------------------ callbacks

  // The x range of a relayout event of the RT tracks: [lo, hi], null for an autorange,
  // or no update when the event does not change the x axis (a y zoom, an autosize).
  function xRange(relayout) {
    if (!relayout) {
      return undefined;
    }
    var lo = null;
    var hi = null;
    var auto = false;
    Object.keys(relayout).forEach(function (k) {
      var m = /^xaxis(\d*)\.(range\[0\]|range\[1\]|range|autorange)$/.exec(k);
      if (!m) {
        return;
      }
      var v = relayout[k];
      if (m[2] === "range[0]") {
        lo = Number(v);
      } else if (m[2] === "range[1]") {
        hi = Number(v);
      } else if (m[2] === "range" && Array.isArray(v) && v.length === 2) {
        lo = Number(v[0]);
        hi = Number(v[1]);
      } else if (m[2] === "autorange" && v) {
        auto = true;
      }
    });
    if (lo !== null && hi !== null && isFinite(lo) && isFinite(hi) && hi > lo) {
      return [lo, hi];
    }
    return auto ? null : undefined;
  }

  window.dash_clientside = Object.assign({}, window.dash_clientside, {
    mvqc: {
      // Each run of an experiment has its own QC page.
      run: function (value, key) {
        if (!key || !value || value === key.run) {
          return NO();
        }
        return key.base + "qc?" + new URLSearchParams({ run: value }).toString();
      },

      xrange: function (relayout, previous) {
        var r = xRange(relayout);
        if (r === undefined) {
          return NO();
        }
        var before = previous ? previous.range : null;
        if (
          (r === null && before === null && previous) ||
          (r && before && r[0] === before[0] && r[1] === before[1])
        ) {
          return NO();
        }
        return { range: r, ts: Date.now() };
      },

      // A click on the isolation-window chart shows that window's MS2 TIC.
      pickWindow: function (click) {
        if (!click || !click.points || !click.points.length) {
          return [NO(), NO()];
        }
        var custom = click.points[0].customdata;
        if (!custom || custom[0] === undefined || custom[0] === null) {
          return [NO(), NO()];
        }
        return ["2", String(Math.round(Number(custom[0])))];
      },
    },
  });
})();
