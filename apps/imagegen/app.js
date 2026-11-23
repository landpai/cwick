/* cwick app: two-step image + video generation studio.
 *
 * Step 1 calls the two image endpoints on this same origin:
 *   /v1/images/edits         prompt + uploaded references (face/outfit/pose)
 *   /v1/images/generations   plain text prompt
 * The image endpoints are synchronous and return one base64 PNG. While a
 * render is in flight, GET /v1/images/progress reports the live pipeline phase
 * (and the DiT step counter), which drives the progress panel; the final
 * per-stage timings come from the response's `cwick` object. The first POST
 * opens the image model on demand, so no manual model switch is needed.
 *
 * Step 2 calls the async video resource: POST /v1/videos creates a job (with
 * the step-1 PNG attached as `image`), GET /v1/videos/{id} polls its status
 * and progress, and GET /v1/videos/{id}/content streams the finished mp4.
 * The progress panel is driven by the server's real job.progress field.
 *
 * Both steps share one SIZE_PAIRS table: choosing a row in step 1 fixes the
 * step-1 image canvas and mirrors its aspect-ratio-matched video canvas into
 * step 2. Manual video sizes remain available in a separate `optgroup` but
 * are marked as unpaired. The quick-verify row sets both canvases to 256x256
 * with a 4 s / 4-step clip.
 *
 * All calls are same-origin and rely on the session cookie; a 401 redirects
 * the top frame to login. History is mirrored into the server's per-app
 * sqlite when available, with localStorage as the fallback.
 */
(function () {
  'use strict';

  var $ = function (id) { return document.getElementById(id); };
  var LS = 'cwick.imagegen.v2';
  var S = {
    mode: 'edit',
    files: { face: null, outfit: null, pose: null },
    busy: false, aborter: null, startedAt: 0, timer: 0, last: null,
    modelLoaded: false,
    videoLoaded: false,
    /* step 2 */
    source: null,          /* { name, dataUrl, base64, mime, fromStep1, prompt } */
    vbusy: false, vcancel: false, vjob: null,
    vstartedAt: 0, vtimer: 0, vlast: null
  };

  /* One row drives both render sizes: step 1 posts `image` to an image
   * endpoint, step 2 posts `video` to /v1/videos. Every video size is an exact
   * multiple of H3V_CANVAS_MULTIPLE (32) and inside 256..2560; every image
   * size is inside 256..6272. Aspect ratio is identical across the pair, so
   * the conditioning reference and the generated frames share a framing.
   * The 2K row is admitted by both engines but takes hours at 30 steps. */
  var SIZE_PAIRS = [
    { id: 'verify',        image: '256x256',   video: '256x256',   ratio: '1:1' },
    { id: 'square',        image: '1024x1024', video: '1024x1024', ratio: '1:1' },
    { id: 'portrait34',    image: '1536x2048', video: '768x1024',  ratio: '3:4' },
    { id: 'portrait23',    image: '1024x1536', video: '768x1152',  ratio: '2:3' },
    { id: 'landscape43',   image: '2048x1536', video: '1280x960',  ratio: '4:3' },
    { id: 'landscape169',  image: '2048x1152', video: '1024x576',  ratio: '16:9' },
    { id: 'k2_169',        image: '2560x1440', video: '2560x1440', ratio: '16:9' }
  ];
  var PAIR_BY_ID = {}, PAIR_BY_VIDEO = {};
  SIZE_PAIRS.forEach(function (p) { PAIR_BY_ID[p.id] = p; PAIR_BY_VIDEO[p.video] = p; });
  function selectedPair() { return PAIR_BY_ID[$('size').value] || null; }
  function samePair(id, videoSize) {
    var p = PAIR_BY_ID[id], v = PAIR_BY_VIDEO[videoSize];
    return !!(p && v && p.id === v.id);
  }
  function updatePairHint() {
    var p = selectedPair();
    var hi = $('pairHintImage'), hv = $('pairHintVideo');
    if (hi) hi.textContent = p
      ? '当前档位：图像 ' + p.image + ' → 视频 ' + p.video + '（' + p.ratio +
        '）。切换档位会同步更新第 2 步尺寸。'
      : '请选择一个配对档位。';
    if (hv) hv.textContent = samePair($('size').value, $('vsize').value)
      ? '已与步骤 1 配对：图像 ' + p.image + ' → 视频 ' + p.video + '（' + p.ratio + '）。'
      : '当前是扩展画布，未与步骤 1 配对；选上方「与步骤 1 配对」分组可恢复一一对应。';
  }
  /* `syncVideo` is false only while reacting to a manual video-size change. */
  function applyPair(id, syncVideo) {
    var p = PAIR_BY_ID[id];
    if (!p) return;
    $('size').value = id;
    if (syncVideo !== false) $('vsize').value = p.video;
    updatePairHint();
  }

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
  function sleep(ms) { return new Promise(function (r) { setTimeout(r, ms); }); }
  function fmtClock(ms) {
    var s = Math.floor(ms / 1000), m = Math.floor(s / 60);
    s = s % 60;
    return m + ':' + (s < 10 ? '0' : '') + s;
  }
  function fmtDur(ms) {
    if (!ms && ms !== 0) return '—';
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
  async function resolveModels() {
    var ip = $('modelPill'), vp = $('videoPill');
    try {
      var res = await fetch('/v1/models', { cache: 'no-store' });
      if (authRedirect(res)) return;
      if (!res.ok) throw new Error('HTTP ' + res.status);
      var j = await res.json();
      var data = Array.isArray(j.data) ? j.data : [];
      var img = null, vid = null;
      for (var i = 0; i < data.length; i++) {
        var c = data[i].cwick || {};
        if (c.kind === 'image' && !img) img = data[i];
        if (c.kind === 'video' && !vid) vid = data[i];
      }
      S.modelLoaded = !!(img && img.cwick && img.cwick.loaded);
      S.videoLoaded = !!(vid && vid.cwick && vid.cwick.loaded);
      ip.textContent = S.modelLoaded ? '图像模型已就绪 · MiniMax-H3-Image'
                                     : '图像模型按需加载 · 首次生成会自动热切换';
      ip.className = 'pill ' + (S.modelLoaded ? 'ok' : 'bad');
      vp.textContent = S.videoLoaded ? '视频模型已就绪 · MiniMax-H3'
                                     : '视频模型按需加载 · 首次生成会自动热切换';
      vp.className = 'pill ' + (S.videoLoaded ? 'ok' : 'bad');
    } catch (e) {
      ip.textContent = vp.textContent = '模型状态读取失败';
      ip.className = vp.className = 'pill bad';
    }
  }

  /* ---------------------------------------------------------------- */
  /* step navigation                                                   */
  /* ---------------------------------------------------------------- */
  function setStep(n) {
    document.querySelectorAll('.step').forEach(function (b) {
      var on = b.getAttribute('data-step') === String(n);
      b.classList.toggle('on', on);
      b.setAttribute('aria-selected', on ? 'true' : 'false');
    });
    $('stage1').hidden = n !== 1;
    $('stage2').hidden = n !== 2;
  }

  /* ---------------------------------------------------------------- */
  /* step 1: uploads + generation                                      */
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
    $('refs').style.display = mode === 'edit' ? '' : 'none';
    $('refHint').textContent = mode === 'edit'
      ? '编辑模式至少上传一张参考图（脸/服装/姿势任意组合）；三者都传效果最佳。'
      : '文生图模式不使用参考图；仅按提示词生成，仍为 3:4 竖构图与写实风格。';
  }
  /* Phases 0..7 as published by GET /v1/images/progress. The server names are
   * idle/tokenize/text_encode/pack/dit/vae/png/done; `queued` here is a local
   * stand-in for phase 0 before the first real snapshot arrives. */
  var PHASES = [
    ['idle', '提交请求'], ['tokenize', '提示词分词'], ['text_encode', '文本编码'],
    ['pack', '参考图打包'], ['dit', '扩散生成（DiT）'], ['vae', '图像解码（VAE）'],
    ['png', 'PNG 编码'], ['done', '完成']
  ];
  /* Weighted bar positions: DiT dominates a render, so it owns most of the bar
   * and advances with the real step/steps counter from the snapshot. */
  var PHASE_AT = [6, 12, 20, 26, 30, 90, 97, 100];
  function progressPct(pr) {
    var base = PHASE_AT[pr.phase] || 6;
    if (pr.phase === 4 && pr.steps > 0) {
      base += (PHASE_AT[5] - PHASE_AT[4]) * (pr.step / pr.steps);
    }
    return Math.max(0, Math.min(100, base));
  }
  function applyProgress(pr) {
    var label = PHASES[pr.phase] ? PHASES[pr.phase][1] : '提交请求';
    if (pr.phase === 4 && pr.steps > 0) label += ' · ' + pr.step + '/' + pr.steps;
    $('progressPhase').textContent = label;
    $('barIn').style.width = progressPct(pr).toFixed(1) + '%';
    document.querySelectorAll('#timeline li').forEach(function (li, i) {
      li.className = i < pr.phase ? 'done' : (i === pr.phase ? 'now' : '');
    });
  }
  async function pollProgress() {
    /* Fire-and-forget: the POST below is the long call; each tick just reads a
     * tiny JSON snapshot. A failed poll never affects the render itself. */
    try {
      var res = await fetch('/v1/images/progress', { cache: 'no-store' });
      if (authRedirect(res) || !res.ok) return;
      var pr = await res.json();
      if (pr && !pr.idle) applyProgress(pr);
    } catch (e) { /* transient: keep the elapsed clock running */ }
  }
  function startProgress() {
    $('progress').hidden = false; $('progressNote').hidden = false;
    $('barIn').style.width = '6%'; $('elapsed').textContent = '0:00';
    var tl = $('timeline'); tl.innerHTML = '';
    PHASES.slice(0, 7).forEach(function (p, i) {
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
      pollProgress();
    }, 1000);
    pollProgress();
  }
  function finishProgress(cw) {
    clearInterval(S.timer); S.timer = 0;
    $('spin').style.display = 'none'; $('barIn').style.width = '100%';
    $('progressPhase').textContent = '完成 · ' + fmtDur(cw && cw.total_ms ? cw.total_ms : (Date.now() - S.startedAt));
    $('elapsed').textContent = fmtClock(cw && cw.total_ms ? cw.total_ms : (Date.now() - S.startedAt));
    var map = { tokenize: 'tokenize_ms', text_encode: 'text_encode_ms', pack: 'pack_ms',
      dit: 'dit_ms', vae: 'vae_ms', png: 'png_ms' };
    document.querySelectorAll('#timeline li').forEach(function (li) {
      var key = li.getAttribute('data-phase');
      li.className = 'done';
      var ms = (key === 'idle' || key === 'queued') ? null : (cw ? cw[map[key]] : null);
      var t = li.querySelector('.mono');
      if (t) t.textContent = ms != null ? fmtDur(ms) : '✓';
    });
    $('progressNote').hidden = true;
  }
  function failProgress(msg) {
    clearInterval(S.timer); S.timer = 0; $('spin').style.display = 'none';
    $('progressPhase').textContent = '失败：' + msg;
    $('barIn').style.width = '100%'; $('barIn').style.background = 'var(--red)';
  }
  function resetProgress() {
    clearInterval(S.timer); S.timer = 0; $('progress').hidden = true;
    $('spin').style.display = ''; $('barIn').style.background = '';
  }
  function b64ToBlob(b64, mime) {
    var bin = atob(b64), len = bin.length, bytes = new Uint8Array(len);
    for (var i = 0; i < len; i++) bytes[i] = bin.charCodeAt(i);
    return new Blob([bytes], { type: mime || 'image/png' });
  }

  async function generate() {
    if (S.busy) return;
    var prompt = ($('prompt').value || '').trim();
    if (!prompt) { alert('请先填写提示词。'); return; }
    var files = S.files;
    var count = (files.face ? 1 : 0) + (files.outfit ? 1 : 0) + (files.pose ? 1 : 0);
    if (S.mode === 'edit' && count === 0) {
      alert('编辑模式至少需要上传一张参考图；或切换到纯文生图模式。'); return;
    }
    /* The image POST opens the model on demand (like the video endpoint), so a
     * missing resident model is not an error the user has to pre-empt. */

    S.busy = true;
    $('generate').disabled = true; $('cancel').disabled = false;
    $('download').disabled = true; $('toVideo').disabled = true;
    $('resultImg').hidden = true; $('emptyState').hidden = false;
    S.aborter = typeof AbortController !== 'undefined' ? new AbortController() : null;
    resetProgress(); startProgress();

    var pair = selectedPair();
    if (!pair) { alert('请选择一个有效的配对尺寸档位。'); return; }
    var size = pair.image, seed = parseInt($('seed').value, 10) || 0;
    var url = S.mode === 'edit' ? '/v1/images/edits' : '/v1/images/generations';
    var opts = { method: 'POST', cache: 'no-store' };
    if (S.aborter) opts.signal = S.aborter.signal;
    try {
      var b64 = null, cw = null;
      if (S.mode === 'edit') {
        var fd = new FormData();
        fd.append('prompt', prompt); fd.append('size', size); fd.append('seed', String(seed));
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
      var text = await res.text(), j = null;
      try { j = JSON.parse(text); } catch (e) { j = null; }
      if (!res.ok) throw new Error((j && j.error && j.error.message) || ('HTTP ' + res.status));
      if (!j || !j.data || !j.data[0] || !j.data[0].b64_json) throw new Error('响应中没有图像数据');
      b64 = j.data[0].b64_json; cw = j.cwick || {};
      var blob = b64ToBlob(b64);
      var objectUrl = URL.createObjectURL(blob);
      var img = $('resultImg'); img.src = objectUrl; img.hidden = false;
      $('emptyState').hidden = true; $('download').disabled = false; $('toVideo').disabled = false;
      S.last = { url: objectUrl, b64: b64, bytes: blob.size, size: size,
        pair: pair.id, cw: cw, prompt: prompt, mode: S.mode, at: Date.now(), refs: count };
      $('download').onclick = function () {
        var a = document.createElement('a');
        a.href = S.last.url;
        a.download = 'cwick-image-' + new Date(S.last.at).toISOString().replace(/[:.]/g, '-') + '.png';
        document.body.appendChild(a); a.click(); document.body.removeChild(a);
      };
      finishProgress(cw);
      pushImageHistory(S.last);
      await resolveModels();
    } catch (e) {
      if (e && e.name === 'AbortError') failProgress('已取消等待');
      else failProgress((e && e.message) || String(e));
    } finally {
      S.busy = false; S.aborter = null;
      $('generate').disabled = false; $('cancel').disabled = true;
    }
  }

  /* ---------------------------------------------------------------- */
  /* step 2: source image                                              */
  /* ---------------------------------------------------------------- */
  function fileToDataUrl(file) {
    return new Promise(function (resolve, reject) {
      var r = new FileReader();
      r.onload = function () { resolve(r.result); };
      r.onerror = function () { reject(new Error('读取图片失败')); };
      r.readAsDataURL(file);
    });
  }
  /* The video endpoint shares the server's request-body ceiling, so a full
   * 1536x2048 PNG as a data URL can be rejected. Shrink the reference to a
   * 1024px JPEG: it is only a visual reference (Ref2VA folds it into the text
   * condition as Picture 1, it does not replace the first-frame latent), so a
   * smaller image keeps the POST well under the limit. Returns
   * { dataUrl, base64, mime }. */
  function shrinkImage(dataUrl, maxSide) {
    return new Promise(function (resolve) {
      var img = new Image();
      img.onload = function () {
        try {
          var w = img.naturalWidth || img.width, h = img.naturalHeight || img.height;
          var scale = Math.min(1, (maxSide || 1024) / Math.max(w, h));
          var cw = Math.max(1, Math.round(w * scale)), ch = Math.max(1, Math.round(h * scale));
          var c = document.createElement('canvas');
          c.width = cw; c.height = ch;
          c.getContext('2d').drawImage(img, 0, 0, cw, ch);
          var out = c.toDataURL('image/jpeg', 0.82);
          resolve({ dataUrl: out, base64: String(out).split(',')[1] || '', mime: 'image/jpeg' });
        } catch (e) { resolve(null); }
      };
      img.onerror = function () { resolve(null); };
      img.src = dataUrl;
    });
  }
  async function setSourceFromDataUrl(dataUrl, name, fromStep1, prompt) {
    var small = await shrinkImage(dataUrl, 1024) ||
      { dataUrl: dataUrl, base64: String(dataUrl).split(',')[1] || '', mime: 'image/png' };
    setSource({ name: name, dataUrl: small.dataUrl, base64: small.base64,
      mime: small.mime, fromStep1: fromStep1, prompt: prompt || '' });
  }
  function setSource(src) {
    S.source = src;
    var img = $('sourceImg'), meta = $('sourceMeta'), empty = $('sourceEmpty');
    if (!src) {
      img.hidden = true; meta.hidden = true; empty.hidden = false;
      $('vpreview').querySelector('.empty').hidden = false;
      return;
    }
    img.src = src.dataUrl; img.hidden = false; empty.hidden = true;
    meta.hidden = false;
    $('sourceName').textContent = src.name || '源图';
    $('sourcePrompt').textContent = src.prompt || (src.fromStep1 ? '来自第 1 步' : '本地上传');
  }
  async function useStep1Image() {
    if (!S.last) return;
    /* Re-assert the pair: the user may have switched video sizes by hand
     * while step 1 was rendering. The image that was just generated must
     * carry its matching video canvas into step 2. */
    if (S.last.pair) applyPair(S.last.pair, true);
    await setSourceFromDataUrl('data:image/png;base64,' + S.last.b64,
      '第 1 步生成图像', true, S.last.prompt);
    setStep(2);
  }

  /* ---------------------------------------------------------------- */
  /* step 2: video job                                                 */
  /* ---------------------------------------------------------------- */
  var VPHASES = [
    ['text', '文本条件编码', 5], ['pack', '噪声与 latent 打包', 10],
    ['dit', '扩散采样（DiT）', 80], ['video_vae', '视频解码（VAE）', 92],
    ['audio_vae', '音频解码', 97], ['mux', 'MP4 合成', 100]
  ];
  /* The server reports these exact phase names in job.phase. */
  var VORDER = ['text', 'pack', 'dit', 'video_vae', 'audio_vae', 'mux'];
  function startVProgress() {
    $('vprogress').hidden = false; $('vprogressNote').hidden = false;
    $('vbarIn').style.width = '1%'; $('vbarIn').style.background = '';
    $('velapsed').textContent = '0:00';
    var tl = $('vtimeline'); tl.innerHTML = '';
    VPHASES.forEach(function (p, i) {
      var li = document.createElement('li');
      li.setAttribute('data-phase', p[0]);
      if (i === 0) li.className = 'now';
      li.innerHTML = '<span>' + p[1] + '</span><span class="mono">—</span>';
      tl.appendChild(li);
    });
    $('vprogressPhase').textContent = '任务已创建，等待服务端调度…';
    S.vstartedAt = Date.now();
    clearInterval(S.vtimer);
    S.vtimer = setInterval(function () {
      $('velapsed').textContent = fmtClock(Date.now() - S.vstartedAt);
    }, 1000);
  }
  function updateVProgress(job) {
    /* The server reports the canonical transformer variant that rendered the
     * job; surface it so the Ref2VA default is visible to the user. */
    if (job.variant) $('vcost').textContent = '变体 · ' + job.variant + '（默认）';
    var pct = Math.max(0, Math.min(100, job.progress || 0));
    $('vbarIn').style.width = pct.toFixed(1) + '%';
    var phase = job.phase || job.status;
    var label = phase;
    for (var i = 0; i < VPHASES.length; i++) {
      if (VPHASES[i][0] === phase) { label = VPHASES[i][1]; break; }
    }
    if (job.status === 'queued') label = '排队中';
    if (job.status === 'completed') label = '完成';
    $('vprogressPhase').textContent = label + ' · ' + pct + '%';
    var idx = VORDER.indexOf(phase);
    document.querySelectorAll('#vtimeline li').forEach(function (li, i) {
      li.className = (idx >= 0 && i < idx) ? 'done' : (idx === i ? 'now' : '');
    });
  }
  function finishVProgress(job, cw) {
    clearInterval(S.vtimer); S.vtimer = 0;
    $('vspin').style.display = 'none'; $('vbarIn').style.width = '100%';
    document.querySelectorAll('#vtimeline li').forEach(function (li) { li.className = 'done'; });
    var ms = cw && cw.total_ms ? cw.total_ms : (Date.now() - S.vstartedAt);
    $('vprogressPhase').textContent = '完成 · ' + fmtDur(ms);
    $('velapsed').textContent = fmtClock(ms);
    $('vprogressNote').hidden = true;
  }
  function failVProgress(msg) {
    clearInterval(S.vtimer); S.vtimer = 0; $('vspin').style.display = 'none';
    $('vprogressPhase').textContent = '失败：' + msg;
    $('vbarIn').style.width = '100%'; $('vbarIn').style.background = 'var(--red)';
  }

  async function generateVideo() {
    if (S.vbusy) return;
    var prompt = ($('vprompt').value || '').trim();
    if (!prompt) { alert('请先填写视频提示词。'); return; }
    var useRef = $('useRef').checked;
    if (useRef && !S.source) { alert('已勾选「把源图作为参考」，但没有源图。请先在第一步生成图像或上传一张图片。'); return; }

    S.vbusy = true; S.vcancel = false;
    $('vgenerate').disabled = true; $('vcancel').disabled = false;
    $('vdownload').disabled = true;
    $('resultVideo').hidden = true; $('vempty').hidden = false;
    $('resultVideo').pause();
    resetVProgress(); startVProgress();
    S.vstartedAt = Date.now();

    var steps = parseInt($('vsteps').value, 10);
    if (!steps || steps < 1) steps = 0;   /* 0 = engine default (30) */
    var body = { prompt: prompt, size: $('vsize').value, seconds: parseInt($('vseconds').value, 10),
      seed: parseInt($('vseed').value, 10) || 0 };
    if (steps) body.steps = steps;
    if (useRef && S.source) body.image = S.source.dataUrl;
    try {
      var res = await fetch('/v1/videos', {
        method: 'POST', cache: 'no-store', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });
      if (authRedirect(res)) throw new Error('authentication required');
      var j = null; try { j = await res.json(); } catch (e) { j = null; }
      if (!res.ok) throw new Error((j && j.error && j.error.message) || ('HTTP ' + res.status));
      if (!j || !j.id) throw new Error('响应中没有任务 id');
      var id = j.id;
      S.vjob = j;
      updateVProgress(j);
      CWICK_INFO_UI('video job ' + id + ' created');

      /* poll until done / failed / cancelled */
      var final = null;
      while (!S.vcancel) {
        await sleep(2000);
        var pr = await fetch('/v1/videos/' + encodeURIComponent(id), { cache: 'no-store' });
        if (authRedirect(pr)) throw new Error('authentication required');
        if (!pr.ok) throw new Error('轮询失败 HTTP ' + pr.status);
        var pj = await pr.json();
        S.vjob = pj;
        updateVProgress(pj);
        if (pj.status === 'completed') { final = pj; break; }
        if (pj.status === 'failed') throw new Error((pj.error && pj.error.message) || '生成失败');
      }
      if (S.vcancel) { failVProgress('已取消等待（服务端任务可能仍在运行）'); return; }

      /* fetch the mp4 bytes */
      var cr = await fetch('/v1/videos/' + encodeURIComponent(id) + '/content', { cache: 'no-store' });
      if (authRedirect(cr)) throw new Error('authentication required');
      if (!cr.ok) throw new Error('下载视频失败 HTTP ' + cr.status);
      var cblob = await cr.blob();
      var curl = URL.createObjectURL(cblob);
      var v = $('resultVideo'); v.src = curl; v.hidden = false; $('vempty').hidden = true;
      $('vdownload').disabled = false;
      S.vlast = { id: id, url: curl, bytes: cblob.size, prompt: prompt, at: Date.now(),
        seconds: body.seconds, size: body.size,
        pair: PAIR_BY_VIDEO[body.size] ? PAIR_BY_VIDEO[body.size].id : null,
        steps: steps, ref: !!(useRef && S.source), cw: final.cwick || null };
      $('vdownload').onclick = function () {
        var a = document.createElement('a');
        a.href = S.vlast.url;
        a.download = 'cwick-video-' + S.vlast.id + '.mp4';
        document.body.appendChild(a); a.click(); document.body.removeChild(a);
      };
      finishVProgress(final, final.cwick);
      pushVideoHistory(S.vlast);
      await resolveModels();
    } catch (e) {
      failVProgress((e && e.message) || String(e));
    } finally {
      S.vbusy = false; S.vcancel = false; S.vjob = null;
      $('vgenerate').disabled = false; $('vcancel').disabled = true;
    }
  }
  function resetVProgress() {
    clearInterval(S.vtimer); S.vtimer = 0; $('vprogress').hidden = true;
    $('vspin').style.display = ''; $('vbarIn').style.background = '';
    $('vcost').textContent = '首次请求会热切换到视频模型';
  }
  function CWICK_INFO_UI(msg) { if (window.console && console.info) console.info('[cwick] ' + msg); }

  /* ---------------------------------------------------------------- */
  /* history                                                           */
  /* ---------------------------------------------------------------- */
  async function pushImageHistory(item) {
    var h = await loadImageHistory();
    h.unshift({ at: item.at, prompt: item.prompt, mode: item.mode, refs: item.refs,
      size: item.size, pair: item.pair, bytes: item.bytes,
      ms: item.cw && item.cw.total_ms });
    if (h.length > 12) h = h.slice(0, 12);
    await storePut('images', h); renderImageHistory(h);
  }
  async function loadImageHistory() {
    var h = await storeGet('images'); return Array.isArray(h) ? h : [];
  }
  function renderImageHistory(h) {
    var box = $('historyList');
    if (!h.length) { box.innerHTML = '<p class="muted small">还没有生成记录。</p>'; return; }
    box.innerHTML = h.map(function (x, i) {
      var label = x.mode === 'edit' ? ('编辑 · ' + (x.refs || 0) + ' 图') : '文生图';
      return '<div class="hist-item" data-i="' + i + '">' +
        '<span class="hist-thumb"></span>' +
        '<span class="hist-meta"><b>' + esc(x.prompt) + '</b>' +
        '<small>' + esc(label) + ' · ' + esc(x.size || '') + ' · ' +
        esc(fmtDur(x.ms || 0)) + ' · ' +
        esc(new Date(x.at).toLocaleString('zh-CN', { hour12: false })) + '</small></span></div>';
    }).join('');
  }
  async function pushVideoHistory(item) {
    var h = await loadVideoHistory();
    h.unshift({ at: item.at, id: item.id, prompt: item.prompt, seconds: item.seconds,
      size: item.size, pair: item.pair, steps: item.steps, ref: item.ref, bytes: item.bytes,
      ms: item.cw && item.cw.total_ms });
    if (h.length > 12) h = h.slice(0, 12);
    await storePut('videos', h); renderVideoHistory(h);
  }
  async function loadVideoHistory() {
    var h = await storeGet('videos'); return Array.isArray(h) ? h : [];
  }
  function renderVideoHistory(h) {
    var box = $('vhistoryList');
    if (!h.length) { box.innerHTML = '<p class="muted small">还没有生成记录。</p>'; return; }
    box.innerHTML = h.map(function (x, i) {
      return '<div class="hist-item" data-i="' + i + '">' +
        '<span class="hist-thumb">▷</span>' +
        '<span class="hist-meta"><b>' + esc(x.prompt) + '</b>' +
        '<small>' + esc(x.size) + ' · ' + esc(x.seconds) + 's' +
        (x.steps ? ' · ' + x.steps + '步' : '') +
        ' · ' + esc(fmtDur(x.ms || 0)) +
        ' · ' + esc(fmtBytes(x.bytes)) + (x.ref ? ' · 参考图' : '') + '</small></span>' +
        '<button class="btn sm" data-replay="' + i + '" type="button">播放</button></div>';
    }).join('');
    box.querySelectorAll('[data-replay]').forEach(function (b) {
      b.addEventListener('click', async function () {
        var x = h[parseInt(b.getAttribute('data-replay'), 10)];
        if (!x) return;
        try {
          var r = await fetch('/v1/videos/' + encodeURIComponent(x.id) + '/content', { cache: 'no-store' });
          if (!r.ok) { alert('该视频已不在服务端（任务槽已回收）。'); return; }
          var blob = await r.blob(), u = URL.createObjectURL(blob);
          var v = $('resultVideo'); v.src = u; v.hidden = false; $('vempty').hidden = true;
          $('vdownload').disabled = false;
          S.vlast = { id: x.id, url: u };
          $('vdownload').onclick = function () {
            var a = document.createElement('a'); a.href = S.vlast.url;
            a.download = 'cwick-video-' + S.vlast.id + '.mp4';
            document.body.appendChild(a); a.click(); document.body.removeChild(a);
          };
        } catch (e) { alert('无法读取视频：' + (e.message || e)); }
      });
    });
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
    document.querySelectorAll('.step').forEach(function (b) {
      b.addEventListener('click', function () { setStep(parseInt(b.getAttribute('data-step'), 10)); });
    });
    document.querySelectorAll('.mode').forEach(function (b) {
      b.addEventListener('click', function () { setMode(b.getAttribute('data-mode')); });
    });
    $('size').addEventListener('change', function () { applyPair(this.value, true); });
    $('vsize').addEventListener('change', updatePairHint);
    $('generate').addEventListener('click', generate);
    $('cancel').addEventListener('click', function () { if (S.aborter) S.aborter.abort(); });
    $('toVideo').addEventListener('click', useStep1Image);
    $('sourceFile').addEventListener('change', async function () {
      var f = this.files && this.files[0];
      if (!f) return;
      var du = await fileToDataUrl(f);
      await setSourceFromDataUrl(du, f.name, false, '');
    });
    $('clear').addEventListener('click', function () {
      $('resultImg').hidden = true; $('resultImg').removeAttribute('src');
      $('emptyState').hidden = false; $('download').disabled = true; $('toVideo').disabled = true;
      S.last = null; resetProgress();
    });
    $('clearHistory').addEventListener('click', async function () {
      await storePut('images', []); renderImageHistory([]);
    });
    $('vgenerate').addEventListener('click', generateVideo);
    $('vcancel').addEventListener('click', function () { S.vcancel = true; });
    $('fastVerify').addEventListener('click', function () {
      /* One click sets both steps: the image endpoint gets the same 256x256
       * request and the video endpoint gets 256x256 / 4 s / 4 steps. */
      applyPair('verify', true);
      $('vseconds').value = '4';
      $('vsteps').value = '4';
      $('fastNote').textContent =
        '已设配对档「快速验证」：步骤 1 图像 256×256、步骤 2 视频 256×256 · 最短 4 秒 · 4 步，本机数分钟出片。';
      var bar = document.querySelector('.fastbar');
      if (bar) bar.classList.add('on');
      CWICK_INFO_UI('quick-verify preset: paired 256x256 / 256x256 / 4s / 4 steps');
    });
    $('vclear').addEventListener('click', function () {
      $('resultVideo').pause(); $('resultVideo').hidden = true; $('resultVideo').removeAttribute('src');
      $('vempty').hidden = false; $('vdownload').disabled = true; resetVProgress();
    });
    $('vclearHistory').addEventListener('click', async function () {
      await storePut('videos', []); renderVideoHistory([]);
    });
    $('share').addEventListener('click', share);
    $('closeShare').addEventListener('click', function () {
      var d = $('shareDialog'); if (d.close) d.close(); else d.removeAttribute('open');
    });
  }

  async function boot() {
    bind(); bindUploads(); setMode('edit'); setStep(1);
    /* The step-1 default is portrait34: mirror it to step 2 on first paint so
     * the two selects are never out of sync. */
    applyPair($('size').value, true);
    await storeProbe();
    renderImageHistory(await loadImageHistory());
    renderVideoHistory(await loadVideoHistory());
    await resolveModels();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
})();
