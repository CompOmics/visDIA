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

  // ------------------------------------------------------------------ peptidoforms
  // The twin of widgets.parse_peptidoform and theme.mod_style; the styles come from
  // theme.py through window.MV (written into the page by app.py).

  const TAG = /\[([^\]]*)\]|\(([^)]*)\)/y;
  const CTERM = /-((?:\[[^\]]*\]|\([^)]*\))+)$/;
  const TAGS = /\[([^\]]*)\]|\(([^)]*)\)/g;

  function mvGlobals() {
    return window.MV || { mods: {}, unimod: {}, modFallback: ["#7048e8"], colours: {} };
  }

  function modName(text) {
    const t = String(text).trim();
    const i = t.indexOf(":");
    if (i > 0 && t.slice(0, i).toLowerCase() === "unimod") {
      const name = mvGlobals().unimod[t.slice(i + 1)];
      if (name) {
        return name;
      }
    }
    return t;
  }

  function modStyle(text) {
    const g = mvGlobals();
    const name = modName(text);
    if (g.mods[name]) {
      return { short: g.mods[name][0], colour: g.mods[name][1], name: name };
    }
    let sum = 0;
    for (let k = 0; k < name.length; k++) {
      sum += name.charCodeAt(k);
    }
    const short = name.length <= 6 ? name : name.slice(0, 3).toLowerCase();
    return { short: short, colour: g.modFallback[sum % g.modFallback.length], name: name };
  }

  function tagAt(text, pos) {
    TAG.lastIndex = pos;
    const m = TAG.exec(text);
    return m ? { tag: m[1] !== undefined ? m[1] : m[2], end: TAG.lastIndex } : null;
  }

  function parsePeptidoform(text) {
    let rest = String(text || "");
    const decoy = rest.startsWith("DECOY_");
    if (decoy) {
      rest = rest.slice(6);
    }
    const nterm = [];
    for (;;) {
      const m = tagAt(rest, 0);
      if (!m || rest.charAt(m.end) !== "-") {
        break;
      }
      nterm.push(m.tag);
      rest = rest.slice(m.end + 1);
    }
    const cterm = [];
    const cm = CTERM.exec(rest);
    if (cm) {
      let t;
      TAGS.lastIndex = 0;
      while ((t = TAGS.exec(cm[1])) !== null) {
        cterm.push(t[1] !== undefined ? t[1] : t[2]);
      }
      rest = rest.slice(0, cm.index);
    }
    const residues = [];
    let pos = 0;
    while (pos < rest.length) {
      const m = tagAt(rest, pos);
      if (m) {
        if (residues.length) {
          residues[residues.length - 1].mods.push(m.tag);
        } else {
          nterm.push(m.tag);
        }
        pos = m.end;
        continue;
      }
      residues.push({ aa: rest.charAt(pos), mods: [] });
      pos += 1;
    }
    return { decoy: decoy, nterm: nterm, residues: residues, cterm: cterm };
  }

  window.mvMods = { name: modName, style: modStyle, parse: parsePeptidoform };

  // Number formats shared by the grids.
  function formatValue(v, format) {
    const x = Number(v);
    if (v === null || v === undefined || Number.isNaN(x)) {
      return "";
    }
    switch (format) {
      case "q":
        return x === 0 ? "0" : x < 0.001 ? x.toExponential(2) : x.toFixed(4);
      case "score":
        return x.toFixed(4);
      case "int":
        return Math.round(x).toLocaleString("en-US");
      case "rt":
        return x.toFixed(1);
      case "compact":
        if (Math.abs(x) >= 1e6) {
          return (x / 1e6).toFixed(1) + " M";
        }
        if (Math.abs(x) >= 1e4) {
          return (x / 1e3).toFixed(1) + " k";
        }
        return x.toPrecision(4);
      default:
        return x.toPrecision(4);
    }
  }
  window.mvFormat = formatValue;

  // A q value that never reads as the threshold when it differs from it (widgets.fmt_q).
  function formatQ(x, t) {
    const form = function (extra) {
      return x < 0.001 ? x.toExponential(2 + extra) : x.toFixed(4 + extra);
    };
    let text = x === 0 ? "0" : form(0);
    if (x === 0 || t === undefined || t === null || x === Number(t)) {
      return text;
    }
    for (let extra = 1; extra < 14; extra++) {
      const shown = Number(text);
      if (shown !== Number(t) && (shown <= t) === (x <= t)) {
        return text;
      }
      text = form(extra);
    }
    return String(x);
  }
  window.mvFormatQ = formatQ;

  // ------------------------------------------------------------------ grid cells
  // AG Grid cell renderers shared by the pages (PeptideShaker-style tables).

  const dag = (window.dashAgGridComponentFunctions = window.dashAgGridComponentFunctions || {});

  function h() {
    return window.React.createElement.apply(window.React, arguments);
  }

  function modSpan(key, label, mods) {
    const styles = mods.map(modStyle);
    const names = styles.map(function (s) { return s.name; }).join(", ");
    return h(
      "span",
      {
        key: key,
        className: "mv-pep-modres",
        style: { color: styles[0].colour },
        title: label === "n" || label === "c" ? names : label + ": " + names,
      },
      label,
      h("sup", { className: "mv-pep-tag" }, styles.map(function (s) { return s.short; }).join("+"))
    );
  }

  // A peptidoform drawn like widgets.peptidoform.
  function peptidoformElement(text) {
    const pf = parsePeptidoform(text);
    const kids = [];
    if (pf.decoy) {
      kids.push(h("span", { key: "decoy", className: "mv-pep-decoy" }, "DECOY_"));
    }
    if (pf.nterm.length) {
      kids.push(modSpan("nt", "n", pf.nterm));
      kids.push(h("span", { key: "ntd", className: "mv-pep-term" }, "-"));
    }
    let plain = "";
    pf.residues.forEach(function (r, i) {
      if (r.mods.length) {
        if (plain) {
          kids.push(plain);
          plain = "";
        }
        kids.push(modSpan("r" + i, r.aa, r.mods));
      } else {
        plain += r.aa;
      }
    });
    if (plain) {
      kids.push(plain);
    }
    if (pf.cterm.length) {
      kids.push(h("span", { key: "ctd", className: "mv-pep-term" }, "-"));
      kids.push(modSpan("ct", "c", pf.cterm));
    }
    return h("span", { className: "mv-pep", title: text }, kids);
  }
  window.mvPeptidoformElement = peptidoformElement;

  dag.MvPeptidoform = function (props) {
    return props.value ? peptidoformElement(String(props.value)) : null;
  };

  // Validation mark. cellRendererParams: qField (the row's q column), threshold (or
  // the grid context's threshold), labelField ("label"), spikeField, transferField.
  dag.MvValidation = function (props) {
    const p = (props.colDef && props.colDef.cellRendererParams) || {};
    const row = props.data || {};
    const field = p.qField || (props.colDef && props.colDef.field);
    const q = row[field];
    const ctx = props.context || {};
    const t = ctx.threshold !== undefined && ctx.threshold !== null ? ctx.threshold : p.threshold;
    const shown =
      q === null || q === undefined || Number.isNaN(Number(q)) ? "not set" : formatQ(Number(q), t);
    let kind = "fail";
    let mark = "✕";
    let tip = "does not pass: " + field + " " + shown + (t !== undefined && t !== null ? " > " + t : "");
    if (row[p.labelField || "label"] === "decoy") {
      kind = "decoy";
      mark = "D";
      tip = "decoy; " + field + " " + shown;
    } else if (q !== null && q !== undefined && t !== undefined && t !== null && Number(q) <= Number(t)) {
      kind = "pass";
      mark = "✓";
      tip = "passes: " + field + " " + shown + " ≤ " + t;
    }
    if (p.spikeField && row[p.spikeField]) {
      kind = "spike";
      mark = "E";
      tip = "entrapment spike-in; " + tip;
    }
    if (p.transferField && row[p.transferField]) {
      tip = "match-between-runs transfer; " + tip;
    }
    return h("span", { className: "mv-valid mv-valid-" + kind, title: tip }, mark);
  };

  // In-cell bar with its value (JSparklines). cellRendererParams: scale ("linear",
  // "log10", "neglog10"), min, max, colour, passColour and failColour (with a
  // threshold: q bars), format (see formatValue), width, tip (what the bar encodes).
  dag.MvBar = function (props) {
    const p = (props.colDef && props.colDef.cellRendererParams) || {};
    const v = props.value;
    if (v === null || v === undefined || Number.isNaN(Number(v))) {
      return null;
    }
    const x = Number(v);
    const tr = function (y) {
      if (p.scale === "log10") {
        return Math.log10(Math.max(y, 1e-300));
      }
      if (p.scale === "neglog10") {
        return -Math.log10(Math.max(y, 1e-300));
      }
      return y;
    };
    const lo = tr(p.min !== undefined ? p.min : 0);
    const hi = tr(p.max !== undefined ? p.max : 1);
    const frac = hi === lo ? 0 : Math.min(1, Math.max(0, (tr(x) - lo) / (hi - lo)));
    const ctx = props.context || {};
    const t = ctx.threshold !== undefined && ctx.threshold !== null ? ctx.threshold : p.threshold;
    let fill = p.colour || "var(--mantine-color-indigo-6)";
    if (p.passColour && t !== undefined && t !== null) {
      fill = x <= Number(t) ? p.passColour : p.failColour || "var(--mantine-color-gray-5)";
    }
    const text = formatValue(x, p.format);
    return h(
      "div",
      { className: "mv-spark-cell", title: p.tip ? text + "; " + p.tip : text },
      h(
        "div",
        { className: "mv-spark", style: { width: (p.width || 56) + "px" } },
        h("div", {
          className: "mv-spark-fill",
          style: { width: (100 * frac).toFixed(1) + "%", background: fill },
        })
      ),
      h("span", { className: "mv-spark-text" }, text)
    );
  };

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
