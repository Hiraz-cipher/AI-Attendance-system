const thresholdInput = document.getElementById("eligibilityThreshold");
const eligibilityExportBtn = document.getElementById("eligibilityExportBtn");
const eligibilityBody = document.getElementById("eligibilityBody");
const eligibilitySessionCount = document.getElementById("eligibilitySessionCount");

async function refreshEligibility() {
  const threshold = parseFloat(thresholdInput.value) || 0;
  eligibilityExportBtn.href = `/api/attendance/eligibility/export?threshold=${threshold}`;

  try {
    const res = await fetch(`/api/attendance/eligibility?threshold=${threshold}`);
    const data = await res.json();

    eligibilitySessionCount.textContent = `${data.total_days} session${data.total_days === 1 ? "" : "s"} recorded`;

    if (data.students.length === 0) {
      eligibilityBody.innerHTML = '<tr><td colspan="4" class="log-empty">No registered students yet.</td></tr>';
      return;
    }

    eligibilityBody.innerHTML = data.students.map(s => `
      <tr>
        <td>${escapeHtml(s.name)}</td>
        <td>${s.present_days} / ${s.total_days}</td>
        <td>${s.percentage}%</td>
        <td>
          <span class="status-chip ${s.eligible ? "status-chip--eligible" : "status-chip--not-eligible"}">
            ${s.eligible ? "Eligible" : "Not Eligible"}
          </span>
        </td>
      </tr>
    `).join("");
  } catch (err) {
    console.error("Failed to refresh eligibility:", err);
  }
}

thresholdInput.addEventListener("input", () => {
  clearTimeout(thresholdInput._debounce);
  thresholdInput._debounce = setTimeout(refreshEligibility, 400);
});

refreshEligibility();
setInterval(refreshEligibility, 30000);
