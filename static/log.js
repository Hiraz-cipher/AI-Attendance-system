function todayStr() {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}

const dateInput = document.getElementById("logDate");
const todayBtn = document.getElementById("logTodayBtn");
const nameFilterInput = document.getElementById("logNameFilter");
const logBody = document.getElementById("logBody");

let allRows = [];
let pollTimer = null;

dateInput.value = todayStr();

async function fetchLog() {
  try {
    const res = await fetch(`/api/attendance?date=${dateInput.value}`);
    allRows = await res.json();
    renderLog();
  } catch (err) {
    console.error("Failed to fetch log:", err);
  }
}

function renderLog() {
  const q = nameFilterInput.value.trim().toLowerCase();
  const rows = q ? allRows.filter(r => r.name.toLowerCase().includes(q)) : allRows;

  if (rows.length === 0) {
    const message = allRows.length === 0
      ? `No check-ins recorded for ${dateInput.value}.`
      : "No names match that search.";
    logBody.innerHTML = `<tr><td colspan="3" class="log-empty">${message}</td></tr>`;
    return;
  }

  logBody.innerHTML = rows.map(r => `
    <tr>
      <td>${escapeHtml(r.name)}</td>
      <td>${escapeHtml(r.time)}</td>
      <td><span class="status-chip">Verified</span></td>
    </tr>
  `).join("");
}

function restartPolling() {
  clearInterval(pollTimer);
  // Only poll live for today's date — a past date's log won't change,
  // so re-fetching it every few seconds would just waste requests.
  if (dateInput.value === todayStr()) {
    pollTimer = setInterval(fetchLog, 4000);
  }
}

dateInput.addEventListener("change", () => {
  fetchLog();
  restartPolling();
});

todayBtn.addEventListener("click", () => {
  dateInput.value = todayStr();
  fetchLog();
  restartPolling();
});

nameFilterInput.addEventListener("input", renderLog);

fetchLog();
restartPolling();
