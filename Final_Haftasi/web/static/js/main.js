/* =========================================================
   ARCHISAFE — Main JavaScript
   ========================================================= */

// ── Socket.IO connection ─────────────────────────────────
let socket = null;
try {
  socket = io({ transports: ['websocket', 'polling'] });

  socket.on('connect', () => console.log('[Socket] Connected'));

  socket.on('fall_alert', (data) => {
    showToast('critical', '🚨 Düşme Tespit Edildi!',
      `Kamera: ${data.camera_name} — Güven: ${Math.round(data.confidence * 100)}%`);
    updateAlertBadge();
    playAlarmSound();
  });

  socket.on('ppe_violation', (data) => {
    showToast('high', '⚠️ KKD İhlali',
      `${data.personnel} — Eksik: ${data.missing_ppe.join(', ')}`);
    updateAlertBadge();
  });

  socket.on('personnel_checkin', (data) => {
    showToast('medium', '✅ Personel Girişi',
      `${data.full_name} sahaya giriş yaptı.`);
  });
} catch (e) { /* Socket.IO not loaded */ }

// ── Sidebar toggle ───────────────────────────────────────
const sidebarToggle = document.getElementById('sidebar-toggle');
const sidebar = document.querySelector('.sidebar');
if (sidebarToggle && sidebar) {
  sidebarToggle.addEventListener('click', () => {
    sidebar.classList.toggle('open');
  });
}

// Click outside to close sidebar on mobile
document.addEventListener('click', (e) => {
  if (sidebar && sidebar.classList.contains('open') &&
      !sidebar.contains(e.target) && e.target !== sidebarToggle) {
    sidebar.classList.remove('open');
  }
});

// ── Toast notifications ──────────────────────────────────
function showToast(severity, title, message, duration = 6000) {
  const container = document.getElementById('toast-container')
    || (() => {
      const c = document.createElement('div');
      c.id = 'toast-container';
      document.body.appendChild(c);
      return c;
    })();

  const toast = document.createElement('div');
  toast.className = `toast toast-${severity}`;

  const icons = { critical: '🚨', high: '⚠️', medium: 'ℹ️', low: '✅' };
  toast.innerHTML = `
    <span class="toast-icon">${icons[severity] || 'ℹ️'}</span>
    <div class="toast-body">
      <div class="toast-title">${escHtml(title)}</div>
      <div class="toast-msg">${escHtml(message)}</div>
    </div>
    <button class="toast-close" onclick="this.closest('.toast').remove()">×</button>
  `;

  container.appendChild(toast);
  if (duration > 0) {
    setTimeout(() => { toast.style.opacity = '0'; setTimeout(() => toast.remove(), 300); }, duration);
  }
}

// ── Flash messages → toasts ──────────────────────────────
document.querySelectorAll('.flash-message').forEach(el => {
  const cat = el.dataset.category || 'medium';
  const catMap = { success: 'low', warning: 'high', danger: 'critical', info: 'medium' };
  showToast(catMap[cat] || 'medium', '', el.textContent.trim());
  el.remove();
});

// ── Dropdown menus ───────────────────────────────────────
document.addEventListener('click', (e) => {
  const trigger = e.target.closest('[data-dropdown]');
  document.querySelectorAll('.dropdown-menu.open').forEach(m => {
    if (!trigger || m.id !== trigger.dataset.dropdown) m.classList.remove('open');
  });
  if (trigger) {
    const menu = document.getElementById(trigger.dataset.dropdown);
    if (menu) menu.classList.toggle('open');
  }
});

// ── Modal helpers ─────────────────────────────────────────
function openModal(id) {
  document.getElementById(id)?.classList.add('open');
}
function closeModal(id) {
  document.getElementById(id)?.classList.remove('open');
}

// Close modal on overlay click
document.querySelectorAll('.modal-overlay').forEach(overlay => {
  overlay.addEventListener('click', (e) => {
    if (e.target === overlay) overlay.classList.remove('open');
  });
});

// ── Alert badge updater ───────────────────────────────────
async function updateAlertBadge() {
  try {
    const res = await fetch('/api/alerts/unread');
    const data = await res.json();
    const badges = document.querySelectorAll('#alert-count');
    badges.forEach(b => {
      b.textContent = data.length;
      b.style.display = data.length > 0 ? 'inline' : 'none';
    });
  } catch (_) {}
}
updateAlertBadge();
setInterval(updateAlertBadge, 15000);

// ── Stats auto-refresh ────────────────────────────────────
async function refreshDashboardStats() {
  try {
    const res = await fetch('/api/stats');
    const data = await res.json();
    for (const [key, val] of Object.entries(data)) {
      const el = document.getElementById(`stat-${key}`);
      if (el) el.textContent = val;
    }
  } catch (_) {}
}
if (document.getElementById('stat-active_personnel')) {
  setInterval(refreshDashboardStats, 20000);
}

// ── Alarm sound ───────────────────────────────────────────
let audioCtx = null;
function playAlarmSound() {
  try {
    audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.connect(gain);
    gain.connect(audioCtx.destination);
    osc.type = 'sawtooth';
    osc.frequency.setValueAtTime(880, audioCtx.currentTime);
    osc.frequency.exponentialRampToValueAtTime(440, audioCtx.currentTime + 0.3);
    gain.gain.setValueAtTime(0.3, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.5);
    osc.start();
    osc.stop(audioCtx.currentTime + 0.5);
  } catch (_) {}
}

// ── Confirm delete ────────────────────────────────────────
document.querySelectorAll('[data-confirm]').forEach(btn => {
  btn.addEventListener('click', (e) => {
    if (!confirm(btn.dataset.confirm || 'Emin misiniz?')) e.preventDefault();
  });
});

// ── Auto-submit filter forms ──────────────────────────────
document.querySelectorAll('[data-auto-submit]').forEach(el => {
  el.addEventListener('change', () => el.closest('form')?.submit());
});

// ── PPE checkbox visual toggle ────────────────────────────
document.querySelectorAll('.ppe-checkbox input').forEach(cb => {
  const toggle = () => {
    cb.closest('.ppe-checkbox').classList.toggle('checked', cb.checked);
  };
  toggle();
  cb.addEventListener('change', toggle);
});

// ── Image preview on file input ───────────────────────────
document.querySelectorAll('[data-preview]').forEach(input => {
  input.addEventListener('change', (e) => {
    const target = document.getElementById(input.dataset.preview);
    if (!target) return;
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = ev => target.src = ev.target.result;
      reader.readAsDataURL(file);
    }
  });
});

// ── Utility ──────────────────────────────────────────────
function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// Expose for inline usage
window.openModal = openModal;
window.closeModal = closeModal;
window.showToast  = showToast;
