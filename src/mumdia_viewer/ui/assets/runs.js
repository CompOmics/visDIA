/* Across-runs page (xr-) and condition-ratio page (cr-): browser callbacks.
   Functions live in window.dash_clientside.mvr. */
(function () {
  "use strict";

  function conditionOf(stored, suggest, run) {
    var v = stored && typeof stored[run] === "string" ? stored[run].trim() : "";
    if (v) return v;
    return suggest && suggest[run] !== undefined ? suggest[run] : "";
  }

  /* "a:b" written the other way round; a bare "r" becomes "1:r" (ratios.flip_ratio). */
  function flipRatio(v) {
    var t = v === null || v === undefined ? "" : String(v).trim();
    if (!t) return t;
    var i = t.search(/[:/]/);
    if (i >= 0) return t.slice(i + 1).trim() + ":" + t.slice(0, i).trim();
    if (/^[0-9]*\.?[0-9]+$/.test(t)) return "1:" + t;
    return t;
  }

  window.dash_clientside = Object.assign({}, window.dash_clientside, {
    mvr: {
      /* The condition under each run name: the stored choice, else the suggestion. The
         page's own store triggers it (the shared store mv-conditions is read as a state,
         so that the callback never fires on another page). */
      conditions: function (suggest, stored, ids) {
        return (ids || []).map(function (id) {
          return conditionOf(stored, suggest, id.run);
        });
      },

      /* The expected ratios of the benchmark design into the species inputs. */
      design: function (n, ids, design) {
        if (!n) return window.dash_clientside.no_update;
        return (ids || []).map(function (id) {
          return design && design[id.sp] !== undefined ? design[id.sp] : "";
        });
      },

      /* Clear every expected ratio. */
      clear: function (n, ids) {
        if (!n) return window.dash_clientside.no_update;
        return (ids || []).map(function () {
          return "";
        });
      },

      /* The species inputs into the store {pair: [A, B], values: {suffix: "a:b"}}: the
         ratios are kept with the conditions they were entered for (ratios.stored_values). */
      expected: function (values, ids, stored, a, b) {
        var old = stored && stored.values ? stored.values : {};
        var flip = stored && stored.pair && stored.pair[0] === b && stored.pair[1] === a && a !== b;
        var out = {};
        Object.keys(old).forEach(function (k) {
          out[k] = flip ? flipRatio(old[k]) : old[k];
        });
        (ids || []).forEach(function (id, i) {
          var v = values && values[i] !== undefined && values[i] !== null ? String(values[i]) : "";
          if (v.trim()) out[id.sp] = v.trim();
          else delete out[id.sp];
        });
        return { pair: [a, b], values: out };
      },

      /* Swap the two conditions; the expected ratios describe the samples, so each
         "a:b" becomes "b:a". */
      swap: function (n, a, b, values) {
        var nu = window.dash_clientside.no_update;
        if (!n) return [nu, nu, (values || []).map(function () { return nu; })];
        return [b, a, (values || []).map(flipRatio)];
      },
    },
  });
})();
