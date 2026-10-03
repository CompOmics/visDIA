/* Quant QC page: the condition editor, the threshold pill, the heatmap row mark and the
   address of the selected protein group. */

(function () {
  const NO = function () {
    return window.dash_clientside.no_update;
  };

  function triggered() {
    const ctx = window.dash_clientside.callback_context || {};
    return (ctx.triggered || []).map(function (t) {
      return String(t.prop_id || "");
    });
  }

  function clean(value) {
    return typeof value === "string" ? value.trim() : "";
  }

  // The colour of each condition: the palette in the order of the first run of each
  // condition (quant_figures.condition_colours does the same on the server).
  function colours(conditions, palette) {
    const out = {};
    let n = 0;
    conditions.forEach(function (c) {
      if (c && !(c in out)) {
        out[c] = palette[n % palette.length];
        n += 1;
      }
    });
    return out;
  }

  window.dash_clientside = Object.assign({}, window.dash_clientside, {
    mvq: {
      // The editor and the store mv-conditions ({run name: condition}, localStorage)
      // follow each other. An empty or missing value of a run means its suggestion.
      // Typing writes the store (the inputs keep what was typed); the reset button
      // removes this result set's runs from the store; the first call and a store
      // changed elsewhere set the inputs.
      conditions: function (values, reset, suggest, stored, ids, palette) {
        const runs = (ids || []).map(function (i) {
          return i.run;
        });
        suggest = suggest || {};
        palette = palette && palette.length ? palette : ["#4c6ef5"];
        const base = stored && typeof stored === "object" ? stored : {};
        const fired = triggered();
        let store = NO();
        let inputs = runs.map(NO);
        let current;
        if (fired.some(function (p) { return p.indexOf("qq-cond-reset") === 0; })) {
          const next = Object.assign({}, base);
          runs.forEach(function (r) {
            delete next[r];
          });
          store = next;
          current = runs.map(function (r) {
            return suggest[r] || "";
          });
          inputs = current.slice();
        } else if (fired.some(function (p) { return p.indexOf('"type":"qq-cond"') >= 0; })) {
          const next = Object.assign({}, base);
          runs.forEach(function (r, i) {
            next[r] = clean((values || [])[i]) || suggest[r] || "";
          });
          store = next;
          current = runs.map(function (r) {
            return next[r];
          });
        } else {
          current = runs.map(function (r) {
            return clean(base[r]) || suggest[r] || "";
          });
          inputs = current.slice();
        }
        const map = colours(current, palette);
        const styles = current.map(function (c) {
          return { background: map[c] || "var(--mantine-color-gray-5)" };
        });
        const effective = store === NO() ? base : store;
        const own = runs.some(function (r) {
          const v = clean(effective[r]);
          return v && v !== (suggest[r] || "");
        });
        const n = Object.keys(map).length;
        const text =
          (own ? "Your conditions, kept in this browser" : "Suggested from the mzML file names") +
          " · " +
          n +
          (n === 1 ? " condition" : " conditions");
        return [store, inputs, styles, text];
      },

      pill: function (t) {
        const x = Number(t);
        if (!(x > 0 && x < 1)) {
          return NO();
        }
        const label = x < 0.001 ? x.toExponential(0) : String(x);
        return "precursor_q · pg_q_value ≤ " + label;
      },

      // Move the frame of the selected row of the LFQ heatmap without redrawing it.
      markRow: function (group, figure) {
        if (!figure || !figure.layout || !figure.data || !figure.data.length) {
          return NO();
        }
        const ys = figure.data[0].y || [];
        const i = Array.prototype.indexOf.call(ys, group);
        const shapes = (figure.layout.shapes || []).filter(function (s) {
          return s.name !== "selected";
        });
        if (i >= 0) {
          const pad = Math.max(0.5, ys.length / 300);
          shapes.push({
            type: "rect",
            xref: "paper",
            yref: "y",
            x0: -0.012,
            x1: 1.0,
            y0: i - 0.5 - pad,
            y1: i + 0.5 + pad,
            line: { color: (window.MV && window.MV.colours && window.MV.colours.target) || "#4263eb", width: 1.5 },
            fillcolor: "rgba(0,0,0,0)",
            name: "selected",
          });
        }
        const layout = Object.assign({}, figure.layout, { shapes: shapes });
        return Object.assign({}, figure, { layout: layout });
      },

      // The centre colour of the relative heatmap follows the theme (the template swap
      // of the shell does not touch colour scales).
      heatScheme: function (scheme, figure) {
        if (!figure || !figure.data || !figure.data.length || figure.data[0].meta !== "qq-relative") {
          return NO();
        }
        let dark = scheme === "dark";
        if (scheme !== "dark" && scheme !== "light") {
          dark = !!(window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches);
        }
        const centre = dark ? "#3a3b3d" : "#f1f3f5";
        const scale = (figure.data[0].colorscale || []).map(function (stop) {
          return stop[0] === 0.5 ? [0.5, centre] : stop;
        });
        if (JSON.stringify(scale) === JSON.stringify(figure.data[0].colorscale)) {
          return NO();
        }
        const trace = Object.assign({}, figure.data[0], { colorscale: scale });
        return Object.assign({}, figure, { data: [trace].concat(figure.data.slice(1)) });
      },

      // The selected group goes into the address (?group=), so a link restores it.
      address: function (group) {
        try {
          const url = new URL(window.location.href);
          if (group) {
            url.searchParams.set("group", group);
          } else {
            url.searchParams.delete("group");
          }
          window.history.replaceState(window.history.state, "", url.pathname + url.search + url.hash);
        } catch (e) {
          /* an address that cannot be rewritten keeps its old query */
        }
        return NO();
      },
    },
  });

  // The threshold pill opens the header's threshold control.
  document.addEventListener("click", function (event) {
    const t = event.target;
    if (t && t.closest && t.closest(".qq-t")) {
      const select = document.getElementById("q-select");
      if (select) {
        select.focus();
        select.click();
      }
    }
  });
})();
