(() => {
  const labels = document.querySelectorAll('[data-camera-status]');
  if (!labels.length) return;
  let busy = false;
  async function check() {
    if (busy) return;
    busy = true;
    try {
      const response = await fetch('/api/camera/status', {cache: 'no-store', signal: AbortSignal.timeout(5000)});
      if (!response.ok || response.redirected) throw new Error('Session expired. Sign in again.');
      const status = await response.json();
      labels.forEach(el => { el.textContent = status.ready ? 'Camera connected' : 'Camera offline'; el.classList.toggle('is-ready', status.ready); });
      document.querySelectorAll('[data-camera-message]').forEach(el => el.textContent = status.ready ? `Camera ${status.source} connected. Face the camera and blink to verify.` : (status.error || 'Waiting for a fresh camera frame…'));
    } catch (error) {
      labels.forEach(el => { el.textContent = 'Connection lost'; el.classList.remove('is-ready'); });
      document.querySelectorAll('[data-camera-message]').forEach(el => el.textContent = error.message);
    } finally { busy = false; }
  }
  document.querySelectorAll('[data-camera-retry]').forEach(button => button.addEventListener('click', () => {
    document.querySelectorAll('.viewfinder__img').forEach(img => { const url = new URL(img.src); url.searchParams.set('retry', Date.now()); img.src = url.href; });
    check();
  }));
  check();
  setInterval(check, 4000);
})();
