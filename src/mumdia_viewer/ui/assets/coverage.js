/* mumdia-viewer: sequence coverage views (ui/coverage.py sends the data, this draws it).
 *
 * Elements: .mvc-bar (the coverage bar), .mvc-lanes (every peptide span in lanes) and
 * .mvc-seq (the sequence text). Each carries data-length, data-states (one character per
 * residue: 0 not covered, 1 covered by peptides that do not pass, 2 passing),
 * data-spans ([start, end, base_peptide_id, passes, sequence, q], 0-based, end exclusive),
 * data-sequence, data-selected, data-group and data-member. Nothing here computes
 * coverage: the states and spans are the server's (data.fasta.protein_coverage).
 *
 * Events for the page: "mv-coverage-pick" {pep, group, el} on a click on a peptide, and
 * "mv-coverage-member" {member, group, pep, el} on a click on another member of the
 * group. A host element with data-cov-req (the id of a dcc.Store) gets a default
 * handling here: a member click writes {group, member, peptide, t} to that store, and a
 * peptide click with data-cov-href opens the identification page at that peptide.
 */

(function () {
  "use strict";

  var LINE = 60;
  var LANES_MAX = 24;

  function esc(text) {
    return String(text)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function pct(n, length) {
    return ((100 * n) / length).toFixed(4) + "%";
  }

  function data(el) {
    if (el.__mvc && el.__mvc.src === el.getAttribute("data-spans") + "|" + el.getAttribute("data-states")) {
      return el.__mvc;
    }
    var spans = [];
    try {
      spans = JSON.parse(el.getAttribute("data-spans") || "[]");
    } catch (e) {
      spans = [];
    }
    el.__mvc = {
      src: el.getAttribute("data-spans") + "|" + el.getAttribute("data-states"),
      length: parseInt(el.getAttribute("data-length") || "0", 10),
      states: el.getAttribute("data-states") || "",
      spans: spans,
      sequence: el.getAttribute("data-sequence") || "",
      group: el.getAttribute("data-group") || "",
      member: el.getAttribute("data-member") || "",
    };
    return el.__mvc;
  }

  function runs(states) {
    var out = [];
    var a = 0;
    for (var i = 1; i <= states.length; i++) {
      if (i === states.length || states.charAt(i) !== states.charAt(a)) {
        out.push([a, i, states.charAt(a)]);
        a = i;
      }
    }
    return out;
  }

  // ------------------------------------------------------------------ the bar

  function drawBar(el) {
    var d = data(el);
    if (!d.length) {
      el.innerHTML = "";
      return;
    }
    var parts = ['<div class="mvc-track">'];
    runs(d.states).forEach(function (r) {
      if (r[2] !== "0") {
        parts.push(
          '<div class="mvc-run mvc-s' + r[2] + '" style="left:' + pct(r[0], d.length) +
            ";width:" + pct(r[1] - r[0], d.length) + '"></div>'
        );
      }
    });
    parts.push('<div class="mvc-sel-layer"></div><div class="mvc-cursor"></div></div>');
    parts.push(
      '<div class="mvc-scale"><span>1</span><span class="mvc-read"></span><span>' +
        d.length.toLocaleString("en-US") + "</span></div>"
    );
    el.innerHTML = parts.join("");
    outline(el, el.getAttribute("data-selected"));
  }

  // Outline the spans of the selected peptide.
  function outline(el, pep) {
    var d = data(el);
    var layer = el.querySelector(".mvc-sel-layer");
    if (!layer) {
      return;
    }
    var parts = [];
    if (pep !== null && pep !== undefined && pep !== "") {
      d.spans.forEach(function (s) {
        if (String(s[2]) === String(pep)) {
          parts.push(
            '<div class="mvc-sel" style="left:' + pct(s[0], d.length) + ";width:" +
              pct(s[1] - s[0], d.length) + '"></div>'
          );
        }
      });
    }
    layer.innerHTML = parts.join("");
  }

  function residueAt(el, clientX) {
    var track = el.querySelector(".mvc-track");
    var d = data(el);
    if (!track || !d.length) {
      return null;
    }
    var box = track.getBoundingClientRect();
    if (box.width <= 0) {
      return null;
    }
    var i = Math.floor(((clientX - box.left) / box.width) * d.length);
    return Math.max(0, Math.min(d.length - 1, i));
  }

  // The spans over a residue: passing first, then the shortest.
  function spansAt(d, i) {
    return d.spans
      .filter(function (s) {
        return s[0] <= i && i < s[1];
      })
      .sort(function (a, b) {
        return b[3] - a[3] || a[1] - a[0] - (b[1] - b[0]);
      });
  }

  function readOut(el, i) {
    var d = data(el);
    var read = el.querySelector(".mvc-read");
    var cursor = el.querySelector(".mvc-cursor");
    if (!read) {
      return;
    }
    if (i === null) {
      read.textContent = "";
      if (cursor) {
        cursor.style.display = "none";
      }
      return;
    }
    var over = spansAt(d, i);
    var passing = over.filter(function (s) {
      return s[3];
    }).length;
    var text = "residue " + (i + 1).toLocaleString("en-US") + " " + (d.sequence.charAt(i) || "") + ": ";
    if (!over.length) {
      text += "not covered";
    } else {
      text +=
        over.length + (over.length === 1 ? " peptide" : " peptides") + " (" + passing + " passing), " +
        over[0][4] + (over.length > 1 ? " ..." : "") + "; click to select";
    }
    read.textContent = text;
    if (cursor) {
      cursor.style.display = "block";
      cursor.style.left = pct(i, d.length);
      cursor.style.width = "max(1px, " + pct(1, d.length) + ")";
    }
  }

  // ------------------------------------------------------------------ lanes

  function drawLanes(el) {
    var d = data(el);
    if (!d.length) {
      el.innerHTML = "";
      return;
    }
    var ends = [];
    var lanes = [];
    var hidden = 0;
    d.spans.forEach(function (s) {
      for (var k = 0; k < ends.length; k++) {
        if (ends[k] <= s[0]) {
          ends[k] = s[1];
          lanes[k].push(s);
          return;
        }
      }
      if (ends.length >= LANES_MAX) {
        hidden += 1;
        return;
      }
      ends.push(s[1]);
      lanes.push([s]);
    });
    var sel = el.getAttribute("data-selected");
    var parts = [];
    lanes.forEach(function (lane) {
      parts.push('<div class="mvc-lane">');
      lane.forEach(function (s) {
        var tip =
          s[4] + " (" + (s[0] + 1) + "-" + s[1] + "); peptide_q_value " +
          (s[5] === null || s[5] === undefined ? "not set" : s[5]) + (s[3] ? " passes" : " does not pass");
        parts.push(
          '<div class="mvc-lane-pep ' + (s[3] ? "mvc-pass" : "mvc-fail") +
            (String(s[2]) === sel ? " is-sel" : "") + '" data-pep="' + esc(s[2]) + '" style="left:' +
            pct(s[0], d.length) + ";width:" + pct(s[1] - s[0], d.length) + '" title="' + esc(tip) + '"></div>'
        );
      });
      parts.push("</div>");
    });
    if (hidden) {
      parts.push('<div class="mvc-more">' + hidden + " more peptide spans are not drawn (the lanes are full).</div>");
    }
    el.innerHTML = parts.join("");
  }

  // ------------------------------------------------------------------ sequence text

  function drawSequence(el) {
    var d = data(el);
    if (!d.sequence) {
      el.innerHTML = "";
      return;
    }
    var sel = el.getAttribute("data-selected");
    var chosen = new Uint8Array(d.sequence.length);
    d.spans.forEach(function (s) {
      if (String(s[2]) === sel) {
        chosen.fill(1, s[0], s[1]);
      }
    });
    var out = [];
    for (var a = 0; a < d.sequence.length; a += LINE) {
      var b = Math.min(a + LINE, d.sequence.length);
      out.push('<div class="mvc-line"><span class="mvc-lnum">' + (a + 1) + "</span><span>");
      var start = a;
      for (var k = a + 1; k <= b; k++) {
        var block = (k - a) % 10 === 0;
        if (
          k === b || block || d.states.charAt(k) !== d.states.charAt(start) || chosen[k] !== chosen[start]
        ) {
          out.push(
            '<span class="mvc-res mvc-s' + d.states.charAt(start) + (chosen[start] ? " is-sel" : "") + '">' +
              esc(d.sequence.slice(start, k)) + "</span>"
          );
          if (block && k < b) {
            out.push(" ");
          }
          start = k;
        }
      }
      out.push("</span></div>");
    }
    el.innerHTML = out.join("");
  }

  // ------------------------------------------------------------------ drawing and events

  var DRAW = { "mvc-bar": drawBar, "mvc-lanes": drawLanes, "mvc-seq": drawSequence };

  function draw(el) {
    for (var cls in DRAW) {
      if (el.classList.contains(cls)) {
        el.__mvcDrawn = el.getAttribute("data-spans") + "|" + el.getAttribute("data-states") + "|" +
          el.getAttribute("data-selected");
        DRAW[cls](el);
        return;
      }
    }
  }

  function drawAll(node) {
    var scope = node && node.querySelectorAll ? node : document;
    var els = scope.querySelectorAll(".mvc-bar, .mvc-lanes, .mvc-seq");
    for (var i = 0; i < els.length; i++) {
      var el = els[i];
      var key = el.getAttribute("data-spans") + "|" + el.getAttribute("data-states") + "|" +
        el.getAttribute("data-selected");
      if (el.__mvcDrawn !== key) {
        draw(el);
      }
    }
    if (node && node.classList && node.matches && node.matches(".mvc-bar, .mvc-lanes, .mvc-seq")) {
      draw(node);
    }
  }

  // Dash inserts and replaces the elements: draw them as they arrive.
  var observer = new MutationObserver(function (records) {
    var seen = false;
    records.forEach(function (r) {
      if (r.type === "attributes" || r.addedNodes.length) {
        seen = true;
      }
    });
    if (seen) {
      drawAll(document);
    }
  });
  function startObserver() {
    observer.observe(document.body, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ["data-spans", "data-states", "data-selected"],
    });
    drawAll(document);
  }
  if (document.body) {
    startObserver();
  } else {
    document.addEventListener("DOMContentLoaded", startObserver);
  }

  function barOf(target) {
    return target && target.closest ? target.closest(".mvc-bar") : null;
  }

  document.addEventListener("mousemove", function (ev) {
    var el = barOf(ev.target);
    if (el) {
      readOut(el, residueAt(el, ev.clientX));
    }
  });
  document.addEventListener(
    "mouseleave",
    function (ev) {
      var el = barOf(ev.target);
      if (el && ev.target === el) {
        readOut(el, null);
      }
    },
    true
  );

  document.addEventListener("click", function (ev) {
    var member = ev.target && ev.target.closest ? ev.target.closest(".mvc-member") : null;
    if (member) {
      ev.preventDefault();
      var strip = member.closest(".mvc-strip, .mvc-full");
      var bar = strip ? strip.querySelector(".mvc-bar") : null;
      document.dispatchEvent(
        new CustomEvent("mv-coverage-member", {
          detail: {
            member: member.getAttribute("data-member"),
            group: member.getAttribute("data-group"),
            pep: bar ? bar.getAttribute("data-selected") : null,
            el: member,
          },
        })
      );
      return;
    }
    var lanePep = ev.target && ev.target.closest ? ev.target.closest(".mvc-lane-pep") : null;
    if (lanePep) {
      var host = lanePep.closest(".mvc-lanes");
      document.dispatchEvent(
        new CustomEvent("mv-coverage-pick", {
          detail: {
            pep: lanePep.getAttribute("data-pep"),
            group: host ? host.getAttribute("data-group") : null,
            el: lanePep,
          },
        })
      );
      return;
    }
    var el = barOf(ev.target);
    if (!el) {
      return;
    }
    var i = residueAt(el, ev.clientX);
    if (i === null) {
      return;
    }
    var over = spansAt(data(el), i);
    if (over.length) {
      document.dispatchEvent(
        new CustomEvent("mv-coverage-pick", {
          detail: { pep: String(over[0][2]), group: el.getAttribute("data-group"), el: el },
        })
      );
    }
  });

  // The page moved the selection: outline the new peptide everywhere.
  function select(pep) {
    var value = pep === null || pep === undefined ? "" : String(pep);
    var els = document.querySelectorAll(".mvc-bar, .mvc-lanes, .mvc-seq");
    for (var i = 0; i < els.length; i++) {
      if (els[i].getAttribute("data-selected") !== value) {
        els[i].setAttribute("data-selected", value);
      }
    }
  }

  // ------------------------------------------------------------------ default handling

  function hostOf(el) {
    return el && el.closest ? el.closest("[data-cov-req]") : null;
  }

  function setProps(id, props) {
    if (window.dash_clientside && window.dash_clientside.set_props) {
      window.dash_clientside.set_props(id, props);
    }
  }

  document.addEventListener("mv-coverage-member", function (ev) {
    var d = ev.detail || {};
    var host = hostOf(d.el);
    var bar = host ? host.querySelector(".mvc-bar") : null;
    if (!host || !bar) {
      return;
    }
    setProps(host.getAttribute("data-cov-req"), {
      data: {
        group: d.group,
        member: d.member,
        peptide: bar.getAttribute("data-selected") || null,
        t: parseFloat(bar.getAttribute("data-threshold")) || null,
        n: Date.now(),
      },
    });
  });

  document.addEventListener("mv-coverage-pick", function (ev) {
    var d = ev.detail || {};
    var host = hostOf(d.el);
    var href = host ? host.getAttribute("data-cov-href") : null;
    var bar = host ? host.querySelector(".mvc-bar") : null;
    if (!href || !d.group || d.pep === null || d.pep === undefined) {
      return;
    }
    var member = bar ? bar.getAttribute("data-member") : "";
    var query = new URLSearchParams();
    if (member) {
      query.set("search", member);
    }
    query.set("group", d.group);
    query.set("peptide", String(d.pep));
    setProps("url", { href: href + "?" + query.toString() });
  });

  window.mvCoverage = { select: select, draw: drawAll, runs: runs, spansAt: spansAt };
})();
