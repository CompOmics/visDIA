/* mumdia-viewer: keys for validation notes on the precursor page (ui/notes.py).
 *
 * A, R and U set the verdict to accepted, rejected or unsure and save it at once. The
 * keys are ignored while typing in a field, and on every other page.
 */

(function () {
  "use strict";

  var KEYS = { a: "accepted", r: "rejected", u: "unsure" };

  document.addEventListener("keydown", function (event) {
    var target = event.target || {};
    var tag = target.tagName || "";
    if (tag === "INPUT" || tag === "TEXTAREA" || target.isContentEditable) {
      return;
    }
    if (event.ctrlKey || event.metaKey || event.altKey) {
      return;
    }
    var verdict = KEYS[(event.key || "").toLowerCase()];
    var save = document.getElementById("pd-note-save");
    if (!verdict || !save || save.disabled || !window.dash_clientside || !window.dash_clientside.set_props) {
      return;
    }
    event.preventDefault();
    window.dash_clientside.set_props("pd-note-verdict", { value: verdict });
    // Save after the control has taken the value.
    setTimeout(function () {
      save.click();
    }, 60);
  });
})();
