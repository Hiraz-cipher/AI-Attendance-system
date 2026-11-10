let currentRange = "today";

const RANGE_LABELS = {
  today: { present: "Present Today", absent: "Absent Today" },
  week:  { present: "Present This Week", absent: "Absent This Week" },
  month: { present: "Present This Month", absent: "Absent This Month" },
};

async function refreshStats() {
  try {
    const res = await fetch(`/api/attendance/stats?range=${currentRange}`);
    const s = await res.json();
    document.getElementById("kpiTotal").textContent = s.total_registered;
    document.getElementById("kpiPresent").textContent = s.present_today;
    document.getElementById("kpiAbsent").textContent = s.absent_today;
    document.getElementById("kpiPercent").textContent = s.attendance_percentage + "%";

    const fabBadge = document.getElementById("fabBadge");
    if (fabBadge) fabBadge.textContent = s.present_today;

    const labels = RANGE_LABELS[s.range] || RANGE_LABELS.today;
    document.getElementById("kpiPresentLabel").textContent = labels.present;
    document.getElementById("kpiAbsentLabel").textContent = labels.absent;
  } catch (err) {
    console.error("Failed to refresh stats:", err);
  }
}

async function refreshLatestLog() {
  try {
    const res = await fetch("/api/attendance");
    const rows = await res.json();
    const body = document.getElementById("logBody");

    if (rows.length === 0) {
      body.innerHTML = '<tr><td colspan="3" class="log-empty">No check-ins yet today.</td></tr>';
      return;
    }

    body.innerHTML = rows.slice(0, 8).map(r => `
      <tr>
        <td>${escapeHtml(r.name)}</td>
        <td>${escapeHtml(r.time)}</td>
        <td><span class="status-chip">Verified</span></td>
      </tr>
    `).join("");
  } catch (err) {
    console.error("Failed to refresh latest log:", err);
  }
}

document.querySelectorAll(".pill").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".pill").forEach(b => b.classList.remove("pill--active"));
    btn.classList.add("pill--active");
    currentRange = btn.dataset.range || "today";
    refreshStats();
  });
});

refreshStats();
refreshLatestLog();
setInterval(refreshStats, 5000);
setInterval(refreshLatestLog, 4000);
