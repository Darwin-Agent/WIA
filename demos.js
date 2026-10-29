(function () {
  "use strict";

  const domains = ["android", "mcp", "os", "search", "swe", "terminal", "web"];
  const labels = { android: "Android", mcp: "MCP", os: "OS", search: "Search", swe: "SWE", terminal: "Terminal", web: "Web" };
  const params = new URLSearchParams(location.search);
  let currentDomain = domains.includes(params.get("domain")) ? params.get("domain") : "android";
  const video = document.getElementById("demoVideo");
  const source = document.getElementById("demoSource");

  function updateLabels() {
    document.getElementById("demoKicker").textContent = labels[currentDomain];
    document.getElementById("demoTitle").textContent = window.I18N.t(`demos.domain.${currentDomain}.title`);
    document.getElementById("demoDesc").textContent = window.I18N.t(`demos.domain.${currentDomain}.desc`);
    document.querySelectorAll(".domain-tab").forEach(button => {
      const selected = button.dataset.domain === currentDomain;
      button.classList.toggle("active", selected);
      button.setAttribute("aria-pressed", String(selected));
      button.setAttribute("aria-controls", "demoVideo");
    });
  }

  function setDomain(domain) {
    if (!domains.includes(domain)) return;
    currentDomain = domain;
    updateLabels();
    const file = `demos/${domain}.mp4`;
    document.getElementById("downloadBtn").href = file;
    if (source.getAttribute("src") !== file) {
      video.pause();
      source.src = file;
      video.load();
    }
  }

  document.getElementById("domainTabs").addEventListener("click", event => {
    const button = event.target.closest(".domain-tab");
    if (!button) return;
    setDomain(button.dataset.domain);
    const url = new URL(location.href);
    url.searchParams.set("domain", currentDomain);
    try { history.replaceState(null, "", url); } catch (_) { /* Local file preview. */ }
  });

  // A language change must not restart a playing video.
  window.addEventListener("i18n:change", updateLabels);
  setDomain(currentDomain);
})();
