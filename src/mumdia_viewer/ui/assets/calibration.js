/* The calibration page (ui/calibration.py): the linked retention-time axis of its plots.
 *
 * The RT error of the accepted identifications and the mass errors across the gradient
 * have apex RT on x. The calibration plot has the observed RT on the y axis of its top
 * panel and the library iRT on the x axis it shares with the residual panel; the
 * fitted curve (layout.meta.curve, monotone) converts one into the other. A zoom on one
 * plot moves the RT axis of the others; a double click takes them back to the range
 * they were built with (layout.meta.home). A figure's echo of a range this script set is
 * ignored, so the plots do not trade updates back and forth.
 */
(function () {
  const NO = function () {
    return window.dash_clientside.no_update;
  };
  const NAMES = ["cal-fit", "cal-err", "cal-mass"];
  const applied = {};

  function meta(fig) {
    return (fig && fig.layout && fig.layout.meta) || {};
  }

  // Whether a figure takes part: the mass figure only across the gradient.
  function linked(name, fig) {
    if (!fig || !fig.layout) {
      return false;
    }
    return name !== "cal-mass" || meta(fig).view === "gradient";
  }

  // The range of a relayout event on `axis`: [lo, hi], "auto", or null.
  function rangeOf(event, axis) {
    if (!event) {
      return null;
    }
    if (event[axis + ".autorange"]) {
      return "auto";
    }
    const lo = event[axis + ".range[0]"];
    const hi = event[axis + ".range[1]"];
    if (lo !== undefined && hi !== undefined) {
      return [Number(lo), Number(hi)];
    }
    const pair = event[axis + ".range"];
    if (Array.isArray(pair) && pair.length === 2) {
      return [Number(pair[0]), Number(pair[1])];
    }
    return null;
  }

  // Linear interpolation on the monotone curve: from = "x" (iRT) to "y" (RT) or back.
  function along(curve, value, from) {
    const xs = from === "x" ? curve.x : curve.y;
    const ys = from === "x" ? curve.y : curve.x;
    const n = xs.length;
    if (!n) {
      return null;
    }
    if (value <= xs[0]) {
      return ys[0];
    }
    if (value >= xs[n - 1]) {
      return ys[n - 1];
    }
    for (let i = 1; i < n; i++) {
      if (xs[i] >= value) {
        const t = xs[i] === xs[i - 1] ? 0 : (value - xs[i - 1]) / (xs[i] - xs[i - 1]);
        return ys[i - 1] + t * (ys[i] - ys[i - 1]);
      }
    }
    return ys[n - 1];
  }

  function convert(fig, range, from) {
    const curve = meta(fig).curve;
    if (!curve || !curve.x || !curve.x.length || !Array.isArray(range)) {
      return null;
    }
    const a = along(curve, range[0], from);
    const b = along(curve, range[1], from);
    return a === null || b === null || !(b > a) ? null : [a, b];
  }

  // The RT range a figure's relayout event sets.
  function rtOf(name, fig, event) {
    if (name !== "cal-fit") {
      return rangeOf(event, "xaxis");
    }
    const y = rangeOf(event, "yaxis");
    if (y !== null) {
      return y;
    }
    const x = rangeOf(event, "xaxis") || rangeOf(event, "xaxis2");
    if (x === "auto" || x === null) {
      return x;
    }
    return convert(fig, x, "x");
  }

  function same(a, b) {
    if (a === "auto" || b === "auto") {
      return a === b;
    }
    if (!Array.isArray(a) || !Array.isArray(b)) {
      return false;
    }
    const span = Math.max(Math.abs(a[1] - a[0]), 1e-9);
    return Math.abs(a[0] - b[0]) < 1e-6 * span && Math.abs(a[1] - b[1]) < 1e-6 * span;
  }

  function home(fig, axis) {
    const range = (meta(fig).home || {})[axis];
    return Array.isArray(range) ? { range: range, autorange: false } : { autorange: true };
  }

  function set(layout, fig, axis, range) {
    const update = range === "auto" ? home(fig, axis) : { range: range, autorange: false };
    layout[axis] = Object.assign({}, layout[axis] || {}, update);
  }

  // A copy of `fig` showing the RT range `rt`.
  function show(name, fig, rt) {
    const layout = Object.assign({}, fig.layout);
    if (name !== "cal-fit") {
      set(layout, fig, "xaxis", rt);
    } else {
      set(layout, fig, "yaxis", rt);
      let irt = rt === "auto" ? "auto" : convert(fig, rt, "y");
      if (Array.isArray(irt)) {
        const pad = 0.04 * (irt[1] - irt[0]);
        irt = [irt[0] - pad, irt[1] + pad];
      }
      if (irt !== null) {
        set(layout, fig, "xaxis", irt);
        set(layout, fig, "xaxis2", irt);
      }
    }
    return Object.assign({}, fig, { layout: layout });
  }

  window.dash_clientside = Object.assign({}, window.dash_clientside, {
    mvc: {
      link: function (fitEvent, errEvent, massEvent, fitFig, errFig, massFig) {
        const out = [NO(), NO(), NO()];
        const trig = (window.dash_clientside.callback_context || {}).triggered || [];
        if (!trig.length) {
          return out;
        }
        let source = null;
        try {
          source = JSON.parse(trig[0].prop_id.split(".")[0]).name;
        } catch (e) {
          return out;
        }
        const figs = { "cal-fit": fitFig, "cal-err": errFig, "cal-mass": massFig };
        const events = { "cal-fit": fitEvent, "cal-err": errEvent, "cal-mass": massEvent };
        if (!linked(source, figs[source])) {
          return out;
        }
        const rt = rtOf(source, figs[source], events[source]);
        if (rt === null) {
          return out;
        }
        if (applied[source] !== undefined && same(applied[source], rt)) {
          delete applied[source];
          return out;
        }
        NAMES.forEach(function (name, i) {
          if (name === source || !linked(name, figs[name])) {
            return;
          }
          applied[name] = rt;
          out[i] = show(name, figs[name], rt);
        });
        return out;
      },
    },
  });
})();
