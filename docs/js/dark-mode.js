(function() {
  var STORAGE_KEY = 'mkdocs-dark-mode';
  var html = document.documentElement;

  function setDark(enabled) {
    if (enabled) {
      html.classList.add('dark-mode');
    } else {
      html.classList.remove('dark-mode');
    }
    try {
      localStorage.setItem(STORAGE_KEY, enabled ? '1' : '0');
    } catch(e) {}
  }

  var stored = (function() {
    try { return localStorage.getItem(STORAGE_KEY); } catch(e) { return null; }
  })();

  if (stored === '1') {
    setDark(true);
  } else if (stored === null && window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
    setDark(true);
  }

  var btn = document.createElement('button');
  btn.className = 'dark-mode-toggle';
  btn.setAttribute('aria-label', 'Toggle dark mode');
  btn.innerHTML = html.classList.contains('dark-mode') ? '\u2600' : '\u263E';
  btn.addEventListener('click', function() {
    var isDark = !html.classList.contains('dark-mode');
    setDark(isDark);
    btn.innerHTML = isDark ? '\u2600' : '\u263E';
  });
  document.body.appendChild(btn);
})();
