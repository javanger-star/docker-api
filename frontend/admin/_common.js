// Shared utilities for admin pages

function getToken() {
  const t = localStorage.getItem('token');
  if (!t) { window.location.href = '/admin/login'; return null; }
  return t;
}

async function api(method, path, body = null, isForm = false) {
  const token = getToken();
  if (!token) return null;

  const opts = {
    method,
    headers: { Authorization: 'Bearer ' + token }
  };
  if (body) {
    if (isForm) {
      opts.body = body; // FormData
    } else {
      opts.headers['Content-Type'] = 'application/json';
      opts.body = JSON.stringify(body);
    }
  }

  const res = await fetch('/api' + path, opts);
  if (res.status === 401) { localStorage.removeItem('token'); window.location.href = '/admin/login'; return null; }
  if (res.status === 204) return null;
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || 'エラーが発生しました');
  return data;
}

function logout() {
  localStorage.removeItem('token');
  window.location.href = '/admin/login';
}

function fmtDate(s) {
  if (!s) return '-';
  return new Date(s).toLocaleString('ja-JP');
}

function renderNav(active) {
  const links = [
    { href: '/admin', label: 'ダッシュボード', key: 'index' },
    { href: '/admin/training', label: '🤖 AIに検品を教える', key: 'training' },
    { href: '/admin/images', label: '画像管理', key: 'images' },
    { href: '/admin/users', label: 'ユーザー管理', key: 'users' },
  ];
  return `
  <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
    <div class="container-fluid">
      <a class="navbar-brand" href="/admin">検品管理システム</a>
      <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navMenu">
        <span class="navbar-toggler-icon"></span>
      </button>
      <div class="collapse navbar-collapse" id="navMenu">
        <ul class="navbar-nav me-auto">
          ${links.map(l => `
            <li class="nav-item">
              <a class="nav-link ${l.key === active ? 'active' : ''}" href="${l.href}">${l.label}</a>
            </li>`).join('')}
        </ul>
        <button class="btn btn-outline-light btn-sm" onclick="logout()">ログアウト</button>
      </div>
    </div>
  </nav>`;
}
