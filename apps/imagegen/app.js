/* cwick app: H5 image generation studio.
 *
 * Served from /apps/imagegen/ by the cwick binary. It calls the two image
 * endpoints on this same origin: /v1/images/edits for prompt + uploaded
 * references (face / outfit / pose), and /v1/images/generations for a plain
 * text prompt. Both return one JSON with a base64 PNG when the render is
 * done; the API has no progress stream, so the progress panel shows elapsed
 * time and the pipeline stages reported in the response's `cwick` timing
 * object once it arrives.
 *
 * All calls are same-origin and rely on the session cookie; a 401 redirects
 * the top frame to the login page. Job history is mirrored into the server's
 * per-app sqlite when available, with localStorage as the fallback.
 */
(function () {
  'use strict';

  var $ = function (id) { return document.getElementById(id); };
  var LS = 'cwick.imagegen.v1';
  var S = {
    mode: 'edit',
    files: { face: null, outfit: null, pose: null },
    busy: false,
    aborter: null,
    startedAt: 0,
    timer: 0,
    last: null,
    modelLoaded: false,
    modelKind: ''
  };

  function esc(s) {
    return String(s == null ? '' : s).replace(/&/g, '&amp;').replace(/</g, '&lt;')
      .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }
  function authRedirect(res) {
    if (!res || res.status !== 401) return false;
    var next = encodeURIComponent(location.pathname + location.search);
    try {
      if (window.top && window.top !== window) window.top.location.href = '/login?next=' + next;
      else location.href = '/login?next=' + next;
    } catch (e) { location.href = '/login?next=' + next; }
    return true;
  }
  function uid() { return Date.now().toString(36) + Math.random().toString(36).slice(2, 7); }
  function fmtClock(ms) {
    var s = Math.floor(ms / 1000);
    var m = Math.floor(s / 60);
    s = s % 60;
    return m + ':' + (s < 10 ? '0' : '') + s;
  }
  function fmtDur(ms) {
    if (ms >= 60000) return (ms / 60000).toFixed(1) + ' 分';
    return (ms / 1000).toFixed(1) + ' 秒';
  }
  function fmtBytes(n) {
    if (!n) return '0 B';
    var u = ['B', 'KB', 'MB', 'GB'], i = 0;
    while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
    return n.toFixed(i ? 1 : 0) + ' ' + u[i];
  }

  /* ---------------------------------------------------------------- */
  /* storage: server sqlite first, localStorage fallback               */
  /* ---------------------------------------------------------------- */
  var STORE = { app: 'imagegen', ok: null };

  function lsGet(key) {
    try { var j = JSON.parse(localStorage.getItem(LS) || '{}') || {}; return j[key] != null ? j[key] : null; }
    catch (e) { return null; }
  }
  function lsPut(key, value) {
    try { var j = JSON.parse(localStorage.getItem(LS) || '{}') || {}; j[key] = value;
      localStorage.setItem(LS, JSON.stringify(j)); return true; } catch (e) { return false; }
  }
  async function storeFetch(path, opts) {
    var res = await fetch(path, Object.assign({ cache: 'no-store' }, opts || {}));
    if (authRedirect(res)) throw new Error('authentication required');
    return res;
  }
  async function storeProbe() {
    try {
      var res = await storeFetch('/v1/store?app=' + STORE.app);
      if (res.ok) { await res.json(); STORE.ok = true; } else STORE.ok = false;
    } catch (e) { STORE.ok = false; }
    return STORE.ok;
  }
  async function storeGet(key) {
    if (STORE.ok === false) return lsGet(key);
    try {
      var res = await storeFetch('/v1/apps/' + STORE.app + '/state?key=' + encodeURIComponent(key));
      if (!res.ok) { STORE.ok = false; return lsGet(key); }
      var j = await res.json();
      return j && j.value != null ? j.value : null;
    } catch (e) { STORE.ok = false; return lsGet(key); }
  }
  async function storePut(key, value) {
    lsPut(key, value);
    if (STORE.ok === false) return false;
    try {
      var res = await storeFetch('/v1/apps/' + STORE.app + '/state', {
        method: 'PUT', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ key: key, value: value })
      });
      if (!res.ok) { STORE.ok = false; return false; }
      return true;
    } catch (e) { STORE.ok = false; return false; }
  }

  /* ---------------------------------------------------------------- */
  /* model state                                                       */
  /* ---------------------------------------------------------------- */
  async function resolveModel() {
    var pill = $('modelPill');
    try {
      var res = await fetch('/v1/models', { cache: 'no-store' });
      if (authRedirect(res)) return;
      if (!res.ok) throw new Error('HTTP ' + res.status);
      var j = await res.json();
      var data = Array.isArray(j.data) ? j.data : [];
      var img = null;
      for (var i = 0; i < data.length; i++) {
        var c = data[i].cwick || {};
        if (c.kind === 'image') { img = data[i]; break; }
      }
      S.modelLoaded = !!(img && img.cwick && img.cwick.loaded);
      if (S.modelLoaded) {
        pill.textContent = '图像模型已就绪 · MiniMax-H3-Image';
        pill.className = 'pill ok';
      } else {
        pill.textContent = '图像模型未加载 · 请在设置里切换';
        pill.className = 'pill bad';
      }
    } catch (e) {
      pill.textContent = '模型状态读取失败';
      pill.className = 'pill bad';
    }
  }

  /* ---------------------------------------------------------------- */
  /* upload handling                                                   */
  /* ---------------------------------------------------------------- */
  function bindUploads() {
    ['face', 'outfit', 'pose'].forEach(function (slot) {
      var input = document.querySelector('[data-input="' + slot + '"]');
      var thumb = document.querySelector('[data-thumb="' + slot + '"]');
      var name = document.querySelector('[data-name="' + slot + '"]');
      if (!input || !thumb || !name) return;
      input.addEventListener('change', function () {
        var f = input.files && input.files[0];
        S.files[slot] = f || null;
        if (!f) { thumb.innerHTML = '<i>' + (slot === 'face' ? '脸' : slot === 'outfit' ? '装' : '姿') + '</i>';
          name.textContent = '未选择'; return; }
        thumb.innerHTML = '';
        var img = document.createElement('img');
        img.alt = slot;
        img.src = URL.createObjectURL(f);
        thumb.appendChild(img);
        name.textContent = f.name + ' · ' + fmtBytes(f.size);
      });
    });
  }

  function setMode(mode) {
    S.mode = mode;
    document.querySelectorAll('.mode').forEach(function (b) {
      var on = b.getAttribute('data-mode') === mode;
      b.classList.toggle('on', on);
      b.setAttribute('aria-selected', on ? 'true' : 'false');
    });
    var refs = $('refs'), hint = $('refHint');
    refs.style.display = mode === 'edit' ? '' : 'none';
    hint.textContent = mode === 'edit'
      ? '编辑模式至少上传一张参考图（脸/服装/姿势任意组合）；三者都传效果最佳。'
      : '文生图模式不使用参考图；仅按提示词生成，仍为 3:4 竖构图与写实风格。';
  }

  /* ---------------------------------------------------------------- */
  /* progress timeline                                                 */
  /* ---------------------------------------------------------------- */
  var PHASES = [
    ['queued', '提交请求'],
    ['tokenize', '提示词分词'],
    ['text_encode', '文本编码'],
    ['pack', '参考图打包'],
    ['dit', '扩散生成（DiT）'],
    ['vae', '图像解码（VAE）'],
    ['png', 'PNG 编码']
  ];

  function startProgress() {
    $('progress').hidden = false;
    $('progressNote').hidden = false;
    $('barIn').style.width = '6%';
    $('elapsed').textContent = '0:00';
    var tl = $('timeline');
    tl.innerHTML = '';
    PHASES.forEach(function (p, i) {
      var li = document.createElement('li');
      li.setAttribute('data-phase', p[0]);
      if (i === 0) li.className = 'now';
      li.innerHTML = '<span>' + p[1] + '</span><span class="mono">—</span>';
      tl.appendChild(li);
    });
    $('progressPhase').textContent = '已提交，等待本机渲染…';
    S.startedAt = Date.now();
    clearInterval(S.timer);
    S.timer = setInterval(function () {
      $('elapsed').textContent = fmtClock(Date.now() - S.startedAt);
      // Without a server progress stream we can only mark the first stage as
      // running and edge the bar forward slowly; real per-stage times arrive
      // in the response and are filled in by finishProgress().
      var secs = (Date.now() - S.startedAt) / 1000;
      var pct = Math.min(88, 6 + secs / 30);
      $('barIn').style.width = pct.toFixed(1) + '%';
    }, 1000);
  }
  function finishProgress(cw) {
    clearInterval(S.timer);
    S.timer = 0;
    $('spin').style.display = 'none';
    $('barIn').style.width = '100%';
    $('progressPhase').textContent = '完成 · ' + fmtDur(cw && cw.total_ms ? cw.total_ms : (Date.now() - S.startedAt));
    $('elapsed').textContent = fmtClock(cw && cw.total_ms ? cw.total_ms : (Date.now() - S.startedAt));
    var map = {
      tokenize: 'tokenize_ms', text_encode: 'text_encode_ms', pack: 'pack_ms',
      dit: 'dit_ms', vae: 'vae_ms', png: 'png_ms'
    };
    document.querySelectorAll('#timeline li').forEach(function (li) {
      var key = li.getAttribute('data-phase');
      li.className = 'done';
      var ms = key === 'queued' ? null : (cw ? cw[map[key]] : null);
      var t = li.querySelector('.mono');
      if (t) t.textContent = ms != null ? fmtDur(ms) : '✓';
    });
    $('progressNote').hidden = true;
  }
  function failProgress(msg) {
    clearInterval(S.timer); S.timer = 0;
    $('spin').style.display = 'none';
    $('progressPhase').textContent = '失败：' + msg;
    $('barIn').style.width = '100%';
    $('barIn').style.background = 'var(--red)';
  }
  function resetProgress() {
    clearInterval(S.timer); S.timer = 0;
    $('progress').hidden = true;
    $('spin').style.display = '';
    $('barIn').style.background = '';
  }

  /* ---------------------------------------------------------------- */
  /* generation                                                        */
  /* ---------------------------------------------------------------- */
  function b64ToBlob(b64) {
    var bin = atob(b64), len = bin.length, bytes = new Uint8Array(len);
    for (var i = 0; i < len; i++) bytes[i] = bin.charCodeAt(i);
    return new Blob([bytes], { type: 'image/png' });
  }

  async function generate() {
    if (S.busy) return;
    var prompt = ($('prompt').value || '').trim();
    if (!prompt) { alert('请先填写提示词。'); return; }
    var files = S.files;
    var count = (files.face ? 1 : 0) + (files.outfit ? 1 : 0) + (files.pose ? 1 : 0);
    if (S.mode === 'edit' && count === 0) {
      alert('编辑模式至少需要上传一张参考图；或切换到纯文生图模式。');
      return;
    }
    if (!S.modelLoaded) {
      if (!confirm('当前图像模型未加载（本机同一时刻只运行一个模型）。\n请先在设置页切换到 MiniMax-H3-Image 再生成。\n\n仍要继续尝试吗？')) return;
    }

    S.busy = true;
    $('generate').disabled = true;
    $('cancel').disabled = false;
    $('download').disabled = true;
    $('resultImg').hidden = true;
    $('emptyState').hidden = false;
    S.aborter = typeof AbortController !== 'undefined' ? new AbortController() : null;
    resetProgress();
    startProgress();

    var size = $('size').value;
    var seed = parseInt($('seed').value, 10) || 0;
    var url = S.mode === 'edit' ? '/v1/images/edits' : '/v1/images/generations';
    var opts = { method: 'POST', cache: 'no-store' };
    if (S.aborter) opts.signal = S.aborter.signal;

    try {
      var cw = null, b64 = null;
      if (S.mode === 'edit') {
        var fd = new FormData();
        fd.append('prompt', prompt);
        fd.append('size', size);
        fd.append('seed', String(seed));
        ['face', 'outfit', 'pose'].forEach(function (slot) {
          if (files[slot]) fd.append('image', files[slot], files[slot].name);
        });
        opts.body = fd;
      } else {
        opts.headers = { 'Content-Type': 'application/json' };
        opts.body = JSON.stringify({ prompt: prompt, size: size, seed: seed });
      }
      var res = await fetch(url, opts);
      if (authRedirect(res)) throw new Error('authentication required');
      var text = await res.text();
      var j = null;
      try { j = JSON.parse(text); } catch (e) { j = null; }
      if (!res.ok) {
        var msg = (j && j.error && j.error.message) || ('HTTP ' + res.status);
        throw new Error(msg);
      }
      if (!j || !j.data || !j.data[0] || !j.data[0].b64_json) throw new Error('响应中没有图像数据');
      b64 = j.data[0].b64_json;
      cw = j.cwick || {};
      var blob = b64ToBlob(b64);
      var objectUrl = URL.createObjectURL(blob);
      var img = $('resultImg');
      img.src = objectUrl;
      img.hidden = false;
      $('emptyState').hidden = true;
      $('download').disabled = false;
      S.last = { url: objectUrl, size: blob.size, cw: cw, prompt: prompt,
        mode: S.mode, at: Date.now(), refs: count };
      $('download').onclick = function () {
        var a = document.createElement('a');
        a.href = S.last.url;
        a.download = 'cwick-image-' + new Date(S.last.at).toISOString().replace(/[:.]/g, '-') + '.png';
        document.body.appendChild(a); a.click(); document.body.removeChild(a);
      };
      finishProgress(cw);
      pushHistory(S.last);
      await resolveModel();
    } catch (e) {
      if (e && e.name === 'AbortError') failProgress('已取消等待');
      else failProgress((e && e.message) || String(e));
    } finally {
      S.busy = false;
      S.aborter = null;
      $('generate').disabled = false;
      $('cancel').disabled = true;
    }
  }

  /* ---------------------------------------------------------------- */
  /* history                                                           */
  /* ---------------------------------------------------------------- */
  function historyKey() { return 'history'; }
  async function pushHistory(item) {
    var h = await loadHistory();
    h.unshift({ at: item.at, prompt: item.prompt, mode: item.mode,
      refs: item.refs, size: item.size, ms: item.cw && item.cw.total_ms });
    if (h.length > 12) h = h.slice(0, 12);
    await storePut(historyKey(), h);
    renderHistory(h);
  }
  async function loadHistory() {
    var h = await storeGet(historyKey());
    return Array.isArray(h) ? h : [];
  }
  function renderHistory(h) {
    var box = $('historyList');
    if (!h.length) { box.innerHTML = '<p class="muted small">还没有生成记录。</p>'; return; }
    box.innerHTML = h.map(function (x, i) {
      var label = x.mode === 'edit' ? ('编辑 · ' + (x.refs || 0) + ' 图') : '文生图';
      return '<div class="hist-item" data-i="' + i + '">' +
        '<span class="hist-thumb"></span>' +
        '<span class="hist-meta"><b>' + esc(x.prompt) + '</b>' +
        '<small>' + esc(label) + ' · ' + esc(fmtDur(x.ms || 0)) + ' · ' +
        esc(new Date(x.at).toLocaleString('zh-CN', { hour12: false })) + '</small></span></div>';
    }).join('');
  }

  /* ---------------------------------------------------------------- */
  /* share                                                             */
  /* ---------------------------------------------------------------- */
  async function share() {
    var dlg = $('shareDialog'), img = $('qrImg'), urlEl = $('shareUrl'), note = $('shareNote');
    img.hidden = true; img.removeAttribute('src');
    urlEl.textContent = '正在读取服务地址…';
    note.className = 'share-note'; note.textContent = '';
    if (dlg.showModal) dlg.showModal(); else dlg.setAttribute('open', '');
    try {
      var res = await fetch('/v1/share?app=imagegen', { cache: 'no-store' });
      if (authRedirect(res)) return;
      if (!res.ok) throw new Error('HTTP ' + res.status);
      var j = await res.json();
      var url = j.app_url ||
        ((j.lan_urls && j.lan_urls.length) ? j.lan_urls[0] + '/apps/imagegen/index.html' : '');
      if (!j.shareable || !url) {
        urlEl.textContent = location.origin + '/apps/imagegen/index.html';
        note.className = 'share-note bad';
        note.textContent = (j.warning || '当前服务没有可分享的局域网地址。') +
          ' 在这台 Mac 上运行 ./start.sh --lan 后重试。';
        return;
      }
      urlEl.textContent = url;
      img.src = '/v1/share/qr?text=' + encodeURIComponent(url);
      img.hidden = false;
      note.textContent = '手机与这台 Mac 连同一个 Wi-Fi，扫码后即可独立使用；首次打开需要输入登录密码。';
    } catch (e) {
      urlEl.textContent = location.origin + '/apps/imagegen/index.html';
      note.className = 'share-note bad';
      note.textContent = '无法读取 /v1/share：' + (e && e.message ? e.message : e);
    }
  }

  /* ---------------------------------------------------------------- */
  /* bind / boot                                                       */
  /* ---------------------------------------------------------------- */
  function bind() {
    document.querySelectorAll('.mode').forEach(function (b) {
      b.addEventListener('click', function () { setMode(b.getAttribute('data-mode')); });
    });
    $('generate').addEventListener('click', generate);
    $('cancel').addEventListener('click', function () {
      if (S.aborter) S.aborter.abort();
    });
    $('clear').addEventListener('click', function () {
      $('resultImg').hidden = true; $('resultImg').removeAttribute('src');
      $('emptyState').hidden = false; $('download').disabled = true; S.last = null;
      resetProgress();
    });
    $('clearHistory').addEventListener('click', async function () {
      await storePut(historyKey(), []);
      renderHistory([]);
    });
    $('share').addEventListener('click', share);
    $('closeShare').addEventListener('click', function () {
      var d = $('shareDialog');
      if (d.close) d.close(); else d.removeAttribute('open');
    });
  }

  async function boot() {
    bind();
    bindUploads();
    setMode('edit');
    await storeProbe();
    renderHistory(await loadHistory());
    await resolveModel();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
})();
