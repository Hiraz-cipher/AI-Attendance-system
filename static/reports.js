const exportStart = document.getElementById("exportStart");
const exportEnd = document.getElementById("exportEnd");
const exportBtn = document.getElementById("exportBtn");
const baseExportUrl = exportBtn.getAttribute("href");

function updateExportLink() {
  const params = new URLSearchParams();
  if (exportStart.value) params.set("start", exportStart.value);
  if (exportEnd.value) params.set("end", exportEnd.value);
  const qs = params.toString();
  exportBtn.href = qs ? `${baseExportUrl}?${qs}` : baseExportUrl;
}

exportStart.addEventListener("change", updateExportLink);
exportEnd.addEventListener("change", updateExportLink);
