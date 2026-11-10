const MIN_CAPTURES = 5;

const nameInput = document.getElementById("studentName");
const captureBtn = document.getElementById("captureBtn");
const finishBtn = document.getElementById("finishBtn");
const countBadge = document.getElementById("captureCount");
const statusMsg = document.getElementById("statusMsg");

let captureCount = 0;

function setStatus(message, kind) {
  statusMsg.textContent = message;
  statusMsg.className = "status-msg" + (kind ? ` status-msg--${kind}` : "");
}

function updateButtons() {
  const hasName = nameInput.value.trim().length > 0;
  captureBtn.disabled = !hasName;
  finishBtn.disabled = !(hasName && captureCount >= MIN_CAPTURES);
}

nameInput.addEventListener("input", () => {
  updateButtons();
});

captureBtn.addEventListener("click", async () => {
  const name = nameInput.value.trim();
  if (!name) return;

  captureBtn.disabled = true;
  setStatus("Capturing...", null);

  try {
    const res = await fetch("/api/register/capture", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    const data = await res.json();

    if (!res.ok || !data.ok) {
      setStatus(data.error || "Capture failed. Try again.", "error");
    } else {
      captureCount = data.count;
      countBadge.innerHTML = `<span class="status-dot"></span> ${captureCount} / ${MIN_CAPTURES} captured`;
      setStatus(`Captured photo ${captureCount}.`, "success");
    }
  } catch (err) {
    setStatus("Network error — is the server still running?", "error");
  } finally {
    updateButtons();
  }
});

finishBtn.addEventListener("click", async () => {
  const name = nameInput.value.trim();
  if (!name) return;

  finishBtn.disabled = true;
  captureBtn.disabled = true;
  setStatus("Saving student and rebuilding recognition model — this can take a few seconds...", null);

  try {
    const res = await fetch("/api/register/finish", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    const data = await res.json();

    if (!res.ok || !data.ok) {
      setStatus(data.error || "Something went wrong finishing registration.", "error");
      updateButtons();
    } else {
      setStatus(`${name} saved with ${data.photo_count} photos. They can now be recognized on the dashboard.`, "success");
      // Reset for the next student
      captureCount = 0;
      nameInput.value = "";
      countBadge.innerHTML = `<span class="status-dot"></span> 0 / ${MIN_CAPTURES} captured`;
      updateButtons();
    }
  } catch (err) {
    setStatus("Network error — is the server still running?", "error");
    updateButtons();
  }
});

updateButtons();
