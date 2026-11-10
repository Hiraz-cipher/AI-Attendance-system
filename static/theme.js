// ============================================================
// theme.js
// ----------
// Dark mode switch, shared across every page. Pairs with a small inline
// "theme-init" script placed in each page's <head> (see any template) that
// runs BEFORE first paint to set data-theme on <html> from localStorage —
// that inline script prevents a flash of the wrong theme on load; this file
// handles the actual switch interaction once the page is interactive.
//
// localStorage is safe to use here — this is a real deployed website
// running in the user's own browser, not a sandboxed preview environment.
// ============================================================

function getCurrentTheme() {
  return document.documentElement.getAttribute("data-theme") === "dark" ? "dark" : "light";
}

function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  try {
    localStorage.setItem("theme", theme);
  } catch (err) {
    // localStorage can throw in rare browser privacy modes — theme still
    // applies for this page load, it just won't persist across reloads.
  }
  updateSwitches(theme);
  if (typeof window.refreshChartTheme === "function") {
    window.refreshChartTheme();
  }
}

function updateSwitches(theme) {
  const isDark = theme === "dark";
  document.querySelectorAll(".theme-switch").forEach(sw => {
    sw.classList.toggle("is-on", isDark);
    sw.setAttribute("aria-checked", isDark ? "true" : "false");
    const icon = sw.querySelector(".theme-switch__thumb i");
    if (icon) {
      icon.className = isDark ? "fa-solid fa-moon" : "fa-solid fa-sun";
    }
  });
}

document.addEventListener("DOMContentLoaded", () => {
  updateSwitches(getCurrentTheme());
  document.querySelectorAll(".theme-switch").forEach(sw => {
    sw.setAttribute("role", "switch");
    sw.setAttribute("tabindex", "0");
    sw.addEventListener("click", () => {
      applyTheme(getCurrentTheme() === "dark" ? "light" : "dark");
    });
    sw.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        applyTheme(getCurrentTheme() === "dark" ? "light" : "dark");
      }
    });
  });
});
