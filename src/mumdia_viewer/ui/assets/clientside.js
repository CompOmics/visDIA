/* mumdia-viewer client-side callbacks and keyboard shortcuts. */

(function () {
  const NO = function () {
    return window.dash_clientside.no_update;
  };

  function effectiveScheme(scheme) {
    if (scheme === "dark" || scheme === "light") {
      return scheme;
    }
    const dark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
    return dark ? "dark" : "light";
  }

  function fontColour(template) {
    return template && template.layout && template.layout.font && template.layout.font.color;
  }

  function formatCount(n) {
    return n === null || n === undefined ? "-" : Number(n).toLocaleString("en-US");
  }

  function stopLabel(t) {
    if (t < 0.001) {
      return t.toExponential(0).replace("e-", "e-");
    }
    return String(t);
  }

  // Commit of the threshold slider: the header control takes the value once the slider
  // has been still for a moment, so dragging does not recompute every page section.
  let commitTimer = null;

  window.dash_clientside = Object.assign({}, window.dash_clientside, {
    mv: {
      // Every graph gets the Plotly template of the colour scheme. A graph that already
      // has it is left alone, so this also fixes graphs rendered before the stored
      // scheme was known.
      retheme: function (scheme, figures, templates) {
        if (!figures) {
          return [];
        }
        const tpl = templates ? templates[effectiveScheme(scheme)] : null;
        if (!tpl) {
          return figures.map(NO);
        }
        const want = fontColour(tpl);
        return figures.map(function (fig) {
          if (!fig || !fig.layout || fontColour(fig.layout.template) === want) {
            return NO();
          }
          const layout = Object.assign({}, fig.layout, { template: tpl });
          return Object.assign({}, fig, { layout: layout });
        });
      },

      scheme: function (scheme) {
        return effectiveScheme(scheme);
      },

      toggleScheme: function (n, scheme) {
        if (!n) {
          return NO();
        }
        return effectiveScheme(scheme) === "dark" ? "light" : "dark";
      },

      threshold: function (value) {
        const t = parseFloat(value);
        return t > 0 && t < 1 ? t : NO();
      },

      openDrawer: function (n) {
        return n ? true : NO();
      },

      burger: function (opened, navbar) {
        return Object.assign({}, navbar || {}, { collapsed: { mobile: !opened } });
      },

      // The threshold slider of the overview. `data` holds the exact counts at every
      // stop (counted on the server with the engine's q columns). While the slider
      // moves, the cards, the curve marker and the label follow; the header control
      // takes the value when the slider stops.
      slide: function (index, data, figure, selected) {
        const n = (data && data.units ? data.units.length : 4) * 3 + 2;
        if (index === null || index === undefined || !data || !data.stops) {
          return Array(n).fill(NO());
        }
        const i = Math.max(0, Math.min(data.stops.length - 1, Math.round(index)));
        const t = data.stops[i];
        const out = [];
        data.units.forEach(function (u) {
          out.push(formatCount(data.targets[u][i]));
        });
        data.units.forEach(function (u) {
          out.push(data.columns[u] + " ≤ " + stopLabel(t));
        });
        data.units.forEach(function (u) {
          const d = data.decoys[u][i];
          out.push(
            d === null || d === undefined
              ? ""
              : formatCount(d) + (d === 1 ? " decoy passes" : " decoys pass") + " the same cut"
          );
        });
        out.push("q ≤ " + stopLabel(t));
        // Move the threshold marker of the identification curves.
        if (figure && figure.layout) {
          const shapes = (figure.layout.shapes || []).map(function (s) {
            return s.name === "threshold" ? Object.assign({}, s, { x0: t, x1: t }) : s;
          });
          const annotations = (figure.layout.annotations || []).map(function (a) {
            return a.name === "threshold"
              ? Object.assign({}, a, { x: Math.log10(t), text: "q ≤ " + stopLabel(t) })
              : a;
          });
          const layout = Object.assign({}, figure.layout, {
            shapes: shapes,
            annotations: annotations,
          });
          out.push(Object.assign({}, figure, { layout: layout }));
        } else {
          out.push(NO());
        }
        const value = data.values[i];
        if (commitTimer) {
          clearTimeout(commitTimer);
        }
        if (value !== selected) {
          commitTimer = setTimeout(function () {
            window.dash_clientside.set_props("q-select", { value: value });
          }, 450);
        }
        return out;
      },

      // Linear or logarithmic y axis.
      axisType: function (value, figure) {
        if (!figure || !figure.layout) {
          return NO();
        }
        const yaxis = Object.assign({}, figure.layout.yaxis || {}, {
          type: value === "log" ? "log" : "linear",
          autorange: true,
        });
        return Object.assign({}, figure, {
          layout: Object.assign({}, figure.layout, { yaxis: yaxis }),
        });
      },

      // The slider follows the header control.
      sliderIndex: function (t, data) {
        if (!data || !data.stops) {
          return NO();
        }
        let best = 0;
        data.stops.forEach(function (s, i) {
          if (Math.abs(Math.log(s) - Math.log(t)) < Math.abs(Math.log(data.stops[best]) - Math.log(t))) {
            best = i;
          }
        });
        return best;
      },
    },
  });

  // Keyboard shortcuts: arrows step through scans on the precursor page, "/" focuses
  // the search box.
  document.addEventListener("keydown", function (event) {
    const target = event.target || {};
    const tag = target.tagName || "";
    if (tag === "INPUT" || tag === "TEXTAREA" || target.isContentEditable) {
      return;
    }
    if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
      const id = event.key === "ArrowLeft" ? "scan-prev" : "scan-next";
      const button = document.getElementById(id);
      if (button && !button.disabled) {
        button.click();
        event.preventDefault();
      }
    } else if (event.key === "/") {
      const search = document.getElementById("global-search");
      if (search) {
        search.focus();
        event.preventDefault();
      }
    }
  });
})();
