/* ============================================================================
   meridian.js — UI behaviour for Meridian (pure client side).

   Two jobs, both kept entirely in the browser so the single Dash data
   callback is never touched:
     1. Render Lucide icons once the layout is in the DOM.
     2. Instant sidebar -> panel switching (no animation, no server round-trip),
        with a resize nudge so Plotly re-fits charts that were hidden.
   ============================================================================ */
(function () {
    "use strict";

    function renderIcons() {
        if (window.lucide && typeof window.lucide.createIcons === "function") {
            try { window.lucide.createIcons(); } catch (e) { /* not ready yet */ }
        }
    }

    function activate(key) {
        if (!key) return;
        document.querySelectorAll(".nav-item").forEach(function (n) {
            n.classList.toggle("active", n.dataset.panel === key);
        });
        document.querySelectorAll(".panel").forEach(function (p) {
            p.classList.toggle("active", p.dataset.panel === key);
        });
        // Plotly only measures visible containers — nudge it after the switch.
        window.dispatchEvent(new Event("resize"));
    }

    // Event delegation: survives Dash re-renders and works before/after icons
    // have been swapped from <i> placeholders to <svg> elements.
    document.addEventListener("click", function (e) {
        var item = e.target.closest(".nav-item");
        if (item) activate(item.dataset.panel);
    });

    // The Dash renderer mounts React after the scripts run, so poll briefly
    // until the navigation exists, rendering icons as soon as they appear.
    var tries = 0;
    var poll = setInterval(function () {
        renderIcons();
        tries += 1;
        if (document.querySelector(".nav-item svg") || tries > 50) {
            clearInterval(poll);
            window.dispatchEvent(new Event("resize"));
        }
    }, 120);

    window.addEventListener("load", renderIcons);
})();
