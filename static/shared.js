// ============================================================
// Loaded on every page via base.html — clock + sidebar student list.
// Page-specific logic (charts, tables, KPI cards) lives in its own
// file (overview.js, log.js, analytics.js, eligibility.js) and is
// only loaded on the page that needs it.
// ============================================================

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function updateClock() {
  const el = document.getElementById("clock");
  if (el) el.textContent = new Date().toLocaleTimeString([], { hour12: false });
}
updateClock();
setInterval(updateClock, 1000);

async function refreshStudentList() {
  const list = document.getElementById("studentList");
  if (!list) return; // page has no sidebar student list (shouldn't happen, but stay safe)

  try {
    const [attRes, summaryRes] = await Promise.all([
      fetch("/api/attendance"),
      fetch("/api/attendance/summary"),
    ]);
    const todayRows = await attRes.json();
    const summary = await summaryRes.json();

    const presentToday = new Set(todayRows.map(r => r.name));
    const allNames = new Set(summary.map(s => s.name));
    todayRows.forEach(r => allNames.add(r.name));

    if (allNames.size === 0) {
      list.innerHTML = '<li class="sidebar__student-empty">No students yet.</li>';
      return;
    }

    list.innerHTML = [...allNames].sort().map(name => `
      <li>
        <span>${escapeHtml(name)}</span>
        <span class="student-dot ${presentToday.has(name) ? "student-dot--present" : "student-dot--absent"}"></span>
      </li>
    `).join("");
  } catch (err) {
    console.error("Failed to refresh student list:", err);
  }
}

const studentSearchInput = document.getElementById("studentSearch");
if (studentSearchInput) {
  studentSearchInput.addEventListener("input", (e) => {
    const q = e.target.value.trim().toLowerCase();
    document.querySelectorAll("#studentList li").forEach(li => {
      const name = li.textContent.trim().toLowerCase();
      li.style.display = name.includes(q) ? "" : "none";
    });
  });
}

refreshStudentList();
setInterval(refreshStudentList, 8000);
