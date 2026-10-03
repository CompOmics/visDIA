/* mumdia-viewer: the compare page (cmp-*). Client-side callbacks (mvx), the cell
   renderers of the unique tables and their AG Grid theme. */

(function () {
  function NO() {
    return window.dash_clientside.no_update;
  }

  function h() {
    return window.React.createElement.apply(window.React, arguments);
  }

  // In-app navigation, as dcc.Link does it: the router builds the page, no reload.
  function go(url) {
    window.history.pushState({}, "", url);
    window.dispatchEvent(new CustomEvent("_dashprivate_pushstate"));
    window.scrollTo(0, 0);
  }

  function plainClick(ev) {
    return ev.button === 0 && !ev.metaKey && !ev.ctrlKey && !ev.shiftKey && !ev.altKey;
  }

  // ------------------------------------------------------------------ grid cells

  var cells = (window.dashAgGridComponentFunctions = window.dashAgGridComponentFunctions || {});

  // The key of a row as a link to its page. Rows of A navigate inside this app; rows
  // of B (grid context `external`) load B's viewer, which is another Dash app.
  cells.CmpLink = function (p) {
    if (!p.data) {
      return null;
    }
    var params = (p.colDef && p.colDef.cellRendererParams) || {};
    var text = String(p.value || "");
    var body =
      params.peptidoform && window.mvPeptidoformElement ? window.mvPeptidoformElement(text) : text;
    var url = p.data._href;
    if (!url) {
      return h("span", { className: "cmp-cell-text" }, body);
    }
    var external = !!(p.context && p.context.external);
    return h(
      "a",
      {
        className: "cmp-link" + (external ? " cmp-link-b" : ""),
        href: url,
        title: (external ? "Open in B's viewer: " : "Open: ") + text,
        onClick: function (ev) {
          ev.stopPropagation();
          if (external || !plainClick(ev)) {
            return;
          }
          ev.preventDefault();
          go(url);
        },
      },
      body
    );
  };

  // What the other side has for a unique key: its smallest q (above the threshold) as
  // a grey bar, or "no target row".
  cells.CmpOther = function (p) {
    if (!p.data) {
      return null;
    }
    var params = (p.colDef && p.colDef.cellRendererParams) || {};
    var side = params.side || "";
    if (p.value === null || p.value === undefined) {
      return h(
        "span",
        {
          className: "cmp-none",
          title: side + " has no target row of this key (not in its library, or not scored)",
        },
        "no target row"
      );
    }
    var q = Number(p.value);
    var frac = Math.min(1, Math.max(0, -Math.log10(Math.max(q, 1e-300)) / 4));
    var text = window.mvFormatQ ? window.mvFormatQ(q, params.threshold) : String(q);
    return h(
      "div",
      {
        className: "mv-spark-cell",
        title:
          side +
          ": smallest q " +
          text +
          " over " +
          (p.data.other_rows || 0) +
          " target row(s); above the threshold " +
          params.threshold,
      },
      h(
        "div",
        { className: "mv-spark", style: { width: "26px" } },
        h("div", {
          className: "mv-spark-fill",
          style: { width: (100 * frac).toFixed(1) + "%", background: "var(--mantine-color-gray-5)" },
        })
      ),
      h("span", { className: "mv-spark-text cmp-other-q" }, text)
    );
  };

  // ------------------------------------------------------------------ grid theme

  window.dashAgGridFunctions = Object.assign(window.dashAgGridFunctions || {}, {
    mvxTheme: function (base) {
      return base.withParams({
        fontFamily: "inherit",
        fontSize: 12.5,
        headerFontSize: 12,
        headerFontWeight: 600,
        backgroundColor: "var(--cmp-grid-bg)",
        foregroundColor: "var(--cmp-grid-fg)",
        textColor: "var(--cmp-grid-fg)",
        subtleTextColor: "var(--cmp-grid-dim)",
        borderColor: "var(--cmp-grid-border)",
        chromeBackgroundColor: "var(--cmp-grid-chrome)",
        headerBackgroundColor: "var(--cmp-grid-header)",
        headerTextColor: "var(--cmp-grid-header-fg)",
        rowHoverColor: "var(--cmp-grid-hover)",
        selectedRowBackgroundColor: "var(--cmp-grid-hover)",
        accentColor: "var(--cmp-grid-accent)",
        wrapperBorder: false,
        wrapperBorderRadius: 0,
        rowBorder: { style: "solid", width: 1, color: "var(--cmp-grid-rowline)" },
        columnBorder: false,
        headerRowBorder: { style: "solid", width: 1, color: "var(--cmp-grid-border)" },
        spacing: 4,
        cellHorizontalPadding: 8,
        rowHeight: 26,
        headerHeight: 28,
        iconSize: 13,
        headerColumnResizeHandleColor: "transparent",
        browserColorScheme: "inherit",
        tooltipBackgroundColor: "var(--cmp-tip-bg)",
        tooltipTextColor: "var(--cmp-tip-fg)",
        tooltipBorder: false,
      });
    },
  });

  // ------------------------------------------------------------------ callbacks

  window.dash_clientside = Object.assign({}, window.dash_clientside, {
    mvx: {
      // A click on a row of the overlap selects that unit below.
      rowUnit: function (clicks, current) {
        var cc = window.dash_clientside.callback_context;
        var id = cc && cc.triggered_id;
        if (!id || !id.unit || !(clicks || []).some(Boolean) || id.unit === current) {
          return NO();
        }
        return id.unit;
      },

      // The selected unit's overlap row is highlighted.
      activeRow: function (unit, ids) {
        return (ids || []).map(function (id) {
          return "cmp-ov-row" + (id && id.unit === unit ? " cmp-ov-row-active" : "");
        });
      },

      // The unit lives in the address (compare?unit=...), so a link restores it. The
      // address is rewritten in place: the page does not rebuild.
      address: function (unit) {
        if (!unit) {
          return NO();
        }
        var url = new URL(window.location.href);
        if (unit === "precursor") {
          url.searchParams.delete("unit");
        } else {
          url.searchParams.set("unit", unit);
        }
        window.history.replaceState(window.history.state, "", url.toString());
        return unit;
      },
    },
  });
})();
