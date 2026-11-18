/* cwick app: TeachMe — full-duplex voice tutor over /v1/realtime.
 *
 * Served from /apps/teachme/ by the cwick binary. It opens a WebSocket to the
 * resident Sophon-Realtime-32B-Tag pipeline on this same origin (the session
 * cookie authenticates the upgrade), streams microphone PCM in as
 * input_audio_buffer.append events, and plays the model's spoken reply from
 * response.audio.delta. Server VAD turns the conversation hands-free: speech
 * onset commits the buffer and starts a response, and talking over the answer
 * barges in. A push-to-talk mode is offered for noisy rooms.
 *
 * The right-hand panel charts the full-chain breakdown the server attaches to
 * every response.done (ear / prefill / decode / talker / first-audio), plus the
 * process-wide aggregate from GET /v1/realtime/stats.
 */
(function () {
  'use strict';

  var $ = function (id) { return document.getElementById(id); };
  var LS = 'cwick.teachme.v1';
  var MODEL = 'Sophon-Realtime-32B-Tag';
  var OUT_RATE = 24000;

  var S = {
    ws: null, connected: false, connecting: false,
    micOn: false, vad: true, responding: false, speaking: false,
    voice: 'vivian_zh',
    captureCtx: null, playCtx: null, stream: null, proc: null, srcNode: null,
    captureRate: 48000,
    playhead: 0, active: [],
    turn: null,            /* assistant turn being received */
    userClip: null,        /* user voice clip being recorded */
    userStart: 0,
    userStartMs: 0,
    turns: [],             /* persisted conversation */
    seq: 0
  };

  function esc(s) {
    return String(s == null ? '' : s).replace(/&/g, '&amp;').replace(/</g, '&lt;')
      .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }
  function uid() { return Date.now().toString(36) + Math.random().toString(36).slice(2, 7); }
  function now() { return performance.now(); }
  function fmtS(x) { return (x == null || isNaN(x)) ? '—' : x.toFixed(2) + ' s'; }
  function authRedirect(res) {
    if (!res || res.status !== 401) return false;
    var next = encodeURIComponent(location.pathname + location.search);
    try {
      if (window.top && window.top !== window) window.top.location.href = '/login?next=' + next;
      else location.href = '/login?next=' + next;
    } catch (e) { location.href = '/login?next=' + next; }
    return true;
  }

  /* ---------------------------------------------------------------- */
  /* logging + state                                                   */
  /* ---------------------------------------------------------------- */
  function log(msg, hot) {
    var ul = $('log');
    var li = document.createElement('li');
    if (hot) li.className = 'hot';
    var t = new Date().toLocaleTimeString('zh-CN', { hour12: false });
    li.innerHTML = '<span class="t">' + t + '</span><span>' + esc(msg) + '</span>';
    ul.appendChild(li);
    while (ul.children.length > 60) ul.removeChild(ul.firstChild);
    ul.scrollTop = ul.scrollHeight;
  }
  function setConn(state, text) {
    var p = $('connPill');
    p.textContent = text;
    p.className = 'pill ' + (state === 'ok' ? 'ok' : state === 'live' ? 'live' : state === 'bad' ? 'bad' : '');
  }
  function setOrb() {
    var orb = $('mic'), label = $('stateLabel'), hint = $('stateHint');
    orb.classList.remove('idle', 'listening', 'speaking', 'thinking');
    if (!S.connected) {
      orb.classList.add('idle'); $('orbCore').textContent = '🎙';
      label.textContent = '点击开始'; hint.textContent = '首次点击会请求麦克风权限';
    } else if (S.responding && !S.speaking) {
      orb.classList.add('thinking'); $('orbCore').textContent = '…';
      label.textContent = '思考中'; hint.textContent = '模型正在组织回答';
    } else if (S.speaking) {
      orb.classList.add('speaking'); $('orbCore').textContent = '🔊';
      label.textContent = '回答中'; hint.textContent = '可随时插话打断';
    } else if (S.micOn) {
      orb.classList.add('listening'); $('orbCore').textContent = '🎙';
      label.textContent = S.vad ? '聆听中' : '录音中';
      hint.textContent = S.vad ? '直接说话，停顿后自动回答' : '再次点击发送这句话';
    } else {
      orb.classList.add('idle'); $('orbCore').textContent = '🎙';
      label.textContent = '已连接'; hint.textContent = '点击麦克风开始说话';
    }
    $('interrupt').disabled = !S.responding;
    $('hangup').disabled = !S.connected;
  }

  /* ---------------------------------------------------------------- */
  /* base64 PCM helpers                                                */
  /* ---------------------------------------------------------------- */
  function pcm16ToB64(i16) {
    var bytes = new Uint8Array(i16.buffer, i16.byteOffset, i16.byteLength);
    var bin = '', CH = 0x8000;
    for (var i = 0; i < bytes.length; i += CH)
      bin += String.fromCharCode.apply(null, bytes.subarray(i, i + CH));
    return btoa(bin);
  }
  function b64ToFloat(b64) {
    var bin = atob(b64), n = bin.length, bytes = new Uint8Array(n);
    for (var i = 0; i < n; i++) bytes[i] = bin.charCodeAt(i);
    var i16 = new Int16Array(bytes.buffer), f = new Float32Array(i16.length);
    for (var j = 0; j < i16.length; j++) f[j] = i16[j] / 32768;
    return f;
  }

  /* ---------------------------------------------------------------- */
  /* playback: a scheduled queue of 24 kHz buffers                     */
  /* ---------------------------------------------------------------- */
  function ensurePlayCtx() {
    if (S.playCtx) return;
    var AC = window.AudioContext || window.webkitAudioContext;
    try { S.playCtx = new AC({ sampleRate: OUT_RATE }); }
    catch (e) { S.playCtx = new AC(); }
  }
  function playDelta(f32) {
    ensurePlayCtx();
    if (S.playCtx.state === 'suspended') S.playCtx.resume();
    var buf = S.playCtx.createBuffer(1, f32.length, OUT_RATE);
    buf.copyToChannel(f32, 0);
    var src = S.playCtx.createBufferSource();
    src.buffer = buf; src.connect(S.playCtx.destination);
    var startAt = Math.max(S.playCtx.currentTime + 0.03, S.playhead);
    src.start(startAt);
    S.playhead = startAt + buf.duration;
    S.active.push(src);
    src.onended = function () {
      var i = S.active.indexOf(src);
      if (i >= 0) S.active.splice(i, 1);
      if (S.active.length === 0 && !S.responding) { S.speaking = false; setOrb(); }
    };
    if (!S.speaking) { S.speaking = true; setOrb(); }
    if (S.turn) S.turn.pcm.push(f32);
  }
  function flushPlayback() {
    for (var i = 0; i < S.active.length; i++) {
      try { S.active[i].onended = null; S.active[i].stop(); } catch (e) {}
    }
    S.active = [];
    if (S.playCtx) S.playhead = S.playCtx.currentTime;
    S.speaking = false;
  }
  function replayTurn(t) {
    if (!t.pcm || !t.pcm.length) return;
    flushPlayback();
    ensurePlayCtx();
    if (S.playCtx.state === 'suspended') S.playCtx.resume();
    var total = 0; t.pcm.forEach(function (c) { total += c.length; });
    var all = new Float32Array(total), off = 0;
    t.pcm.forEach(function (c) { all.set(c, off); off += c.length; });
    var buf = S.playCtx.createBuffer(1, all.length, OUT_RATE);
    buf.copyToChannel(all, 0);
    var src = S.playCtx.createBufferSource();
    src.buffer = buf; src.connect(S.playCtx.destination);
    S.playhead = S.playCtx.currentTime + buf.duration;
    src.start();
    S.speaking = true; setOrb();
    src.onended = function () { S.speaking = false; setOrb(); };
  }

  /* ---------------------------------------------------------------- */
  /* capture: microphone -> PCM16 append events                        */
  /* ---------------------------------------------------------------- */
  function send(obj) {
    if (S.ws && S.ws.readyState === 1) S.ws.send(JSON.stringify(obj));
  }
  async function startMic() {
    if (S.micOn) return;
    try {
      S.stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true }
      });
    } catch (e) {
      log('麦克风权限被拒绝：' + (e.message || e), true);
      alert('需要麦克风权限才能语音对话。请在浏览器地址栏允许麦克风后重试。');
      return;
    }
    var AC = window.AudioContext || window.webkitAudioContext;
    S.captureCtx = new AC();
    S.captureRate = S.captureCtx.sampleRate;
    if (S.captureCtx.state === 'suspended') await S.captureCtx.resume();
    S.srcNode = S.captureCtx.createMediaStreamSource(S.stream);
    S.proc = S.captureCtx.createScriptProcessor(4096, 1, 1);
    var mute = S.captureCtx.createGain(); mute.gain.value = 0;
    S.srcNode.connect(S.proc); S.proc.connect(mute); mute.connect(S.captureCtx.destination);
    S.proc.onaudioprocess = function (e) {
      if (!S.micOn || !S.connected) return;
      var f = e.inputBuffer.getChannelData(0);
      var rms = 0;
      var i16 = new Int16Array(f.length);
      for (var i = 0; i < f.length; i++) {
        var v = f[i]; rms += v * v;
        var q = Math.max(-1, Math.min(1, v)) * 32767;
        i16[i] = q < 0 ? q - 0.5 : q + 0.5;
      }
      rms = Math.sqrt(rms / f.length);
      $('levelIn').style.width = Math.min(100, rms * 320).toFixed(0) + '%';
      send({ type: 'input_audio_buffer.append', audio: pcm16ToB64(i16) });
    };
    S.micOn = true;
    /* Tell the server our capture rate so it resamples with its own sinc. */
    sendSession();
    setOrb();
    log('麦克风已开启 · ' + S.captureRate + ' Hz');
  }
  function stopMic() {
    S.micOn = false;
    if (S.proc) { S.proc.onaudioprocess = null; try { S.proc.disconnect(); } catch (e) {} S.proc = null; }
    if (S.srcNode) { try { S.srcNode.disconnect(); } catch (e) {} S.srcNode = null; }
    if (S.stream) { S.stream.getTracks().forEach(function (t) { t.stop(); }); S.stream = null; }
    if (S.captureCtx) { try { S.captureCtx.close(); } catch (e) {} S.captureCtx = null; }
    $('levelIn').style.width = '0%';
    setOrb();
  }

  /* ---------------------------------------------------------------- */
  /* session config                                                    */
  /* ---------------------------------------------------------------- */
  function sendSession() {
    var rate = S.captureRate;
    if ([16000, 24000, 44100, 48000].indexOf(rate) < 0) rate = 16000;
    var sess = {
      modalities: ['text', 'audio'],
      voice: S.voice,
      input_audio_format: 'pcm16',
      output_audio_format: 'pcm16',
      input_audio_sample_rate: rate,
      input_audio_transcription: null,
      turn_detection: S.vad ? {
        type: 'server_vad', threshold: 0.5,
        prefix_padding_ms: 300, silence_duration_ms: 500,
        create_response: true, interrupt_response: true
      } : null
    };
    send({ type: 'session.update', session: sess });
  }

  /* ---------------------------------------------------------------- */
  /* conversation rendering                                            */
  /* ---------------------------------------------------------------- */
  function clearEmpty() { var e = $('emptyState'); if (e) e.remove(); }
  function scrollConvo() { var c = $('convo'); c.scrollTop = c.scrollHeight; }
  function addUserClip() {
    clearEmpty();
    var div = document.createElement('div');
    div.className = 'msg user';
    div.innerHTML = '<span class="who">你</span>' +
      '<div class="bubble"><span class="voiceclip"><span class="wave">' +
      '<i style="animation-delay:0s"></i><i style="animation-delay:.15s"></i>' +
      '<i style="animation-delay:.3s"></i><i style="animation-delay:.45s"></i>' +
      '</span><span class="dur">说话中…</span></span></div>';
    $('convo').appendChild(div);
    scrollConvo();
    S.userClip = div; S.userStart = now();
    return div;
  }
  function finishUserClip(ms) {
    if (!S.userClip) return;
    var dur = S.userClip.querySelector('.dur');
    var secs = ms != null ? ms / 1000 : (now() - S.userStart) / 1000;
    if (dur) dur.textContent = '语音 · ' + secs.toFixed(1) + ' s';
    var wave = S.userClip.querySelector('.wave');
    if (wave) wave.innerHTML = '<span style="color:var(--accent)">🎤</span>';
    S.turns.push({ role: 'user', secs: secs, at: Date.now() });
    S.userClip = null;
    persist();
  }
  function addBotBubble() {
    clearEmpty();
    var div = document.createElement('div');
    div.className = 'msg bot';
    div.innerHTML = '<span class="who">智子 Sophon</span>' +
      '<div class="bubble"><span class="txt"></span><span class="cursor"></span></div>' +
      '<div class="meta"><span class="timing"></span></div>';
    $('convo').appendChild(div);
    scrollConvo();
    S.turn = { el: div, text: '', pcm: [], at: Date.now(), cw: null };
    return S.turn;
  }
  function botDelta(piece) {
    if (!S.turn) addBotBubble();
    S.turn.text += piece;
    var txt = S.turn.el.querySelector('.txt');
    txt.textContent = S.turn.text;
    scrollConvo();
  }
  function finishBot(status) {
    if (!S.turn) return;
    var cur = S.turn.el.querySelector('.cursor');
    if (cur) cur.remove();
    var meta = S.turn.el.querySelector('.meta');
    var cw = S.turn.cw || {};
    var timing = meta.querySelector('.timing');
    var bits = [];
    if (cw.first_audio_s != null) bits.push('首音频 ' + cw.first_audio_s.toFixed(2) + 's');
    if (cw.turn_s != null) bits.push('整轮 ' + cw.turn_s.toFixed(2) + 's');
    if (status === 'cancelled') bits.push('已被打断');
    timing.textContent = bits.join(' · ');
    if (S.turn.pcm.length) {
      var btn = document.createElement('button');
      btn.className = 'replay'; btn.type = 'button'; btn.textContent = '▶ 重播';
      var t = S.turn;
      btn.onclick = function () { replayTurn(t); };
      meta.appendChild(btn);
    }
    S.turns.push({ role: 'bot', text: S.turn.text, secs: cw.audio_out_s,
      cw: cw, status: status, at: Date.now() });
    if (S.turns.length > 40) S.turns = S.turns.slice(-40);
    S.turn = null;
    persist();
    scrollConvo();
  }

  /* ---------------------------------------------------------------- */
  /* performance panel                                                 */
  /* ---------------------------------------------------------------- */
  function updatePerf(cw) {
    if (!cw) return;
    $('kFirst').textContent = fmtS(cw.first_audio_s);
    $('kTurn').textContent = fmtS(cw.turn_s);
    $('kDecode').textContent = cw.decode_s > 0 ? (cw.reply_tokens / cw.decode_s).toFixed(2) : '—';
    $('kTalker').textContent = cw.talker_s > 0 ? (cw.audio_out_s / cw.talker_s).toFixed(2) + '×' : '—';
    $('stageTotal').textContent = fmtS(cw.turn_s);
    var stages = { ear: cw.ear_s, prefill: cw.prefill_s, decode: cw.decode_s, talker: cw.talker_s };
    var max = Math.max(cw.ear_s, cw.prefill_s, cw.decode_s, cw.talker_s, 0.001);
    Object.keys(stages).forEach(function (k) {
      var row = document.querySelector('.stage[data-stage="' + k + '"]');
      if (!row) return;
      row.querySelector('.s-in').style.width = (stages[k] / max * 100).toFixed(1) + '%';
      row.querySelector('.s-val').textContent = fmtS(stages[k]);
    });
    $('tPrompt').textContent = cw.prompt_tokens != null ? cw.prompt_tokens : '—';
    $('tAudio').textContent = cw.audio_tokens != null ? cw.audio_tokens : '—';
    $('tReply').textContent = cw.reply_tokens != null ? cw.reply_tokens : '—';
    $('tSpeech').textContent = cw.speech_tokens != null ? cw.speech_tokens : '—';
    $('tIn').textContent = cw.audio_in_s != null ? cw.audio_in_s.toFixed(2) + ' s' : '—';
    $('tOut').textContent = cw.audio_out_s != null ? cw.audio_out_s.toFixed(2) + ' s' : '—';
  }
  async function refreshAggregate() {
    try {
      var res = await fetch('/v1/realtime/stats', { cache: 'no-store' });
      if (authRedirect(res)) return;
      if (!res.ok) throw new Error('HTTP ' + res.status);
      var j = await res.json();
      $('aggPill').textContent = j.model || MODEL;
      $('aggPill').className = 'pill ok';
      $('aSessions').textContent = j.sessions;
      $('aTurns').textContent = j.turns;
      $('aCancel').textContent = j.cancelled;
      $('aFirst').textContent = j.mean_per_turn ? fmtS(j.mean_per_turn.first_audio_s) : '—';
    } catch (e) {
      $('aggPill').textContent = '读取失败';
      $('aggPill').className = 'pill bad';
    }
  }

  /* ---------------------------------------------------------------- */
  /* WebSocket                                                         */
  /* ---------------------------------------------------------------- */
  function wsUrl() {
    var proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    return proto + '//' + location.host + '/v1/realtime?model=' + encodeURIComponent(MODEL);
  }
  function connect() {
    return new Promise(function (resolve, reject) {
      if (S.connected) { resolve(); return; }
      if (S.connecting) { reject(new Error('connecting')); return; }
      S.connecting = true;
      setConn('live', '连接中…');
      var ws;
      try { ws = new WebSocket(wsUrl()); }
      catch (e) { S.connecting = false; reject(e); return; }
      S.ws = ws;
      var opened = false;
      ws.onopen = function () { log('WebSocket 已连接'); };
      ws.onmessage = function (ev) {
        var j;
        try { j = JSON.parse(ev.data); } catch (e) { return; }
        handleEvent(j);
        if (!opened && (j.type === 'session.created' || j.type === 'session.updated')) {
          opened = true; S.connected = true; S.connecting = false;
          setConn('ok', '已连接');
          resolve();
        }
      };
      ws.onerror = function () { log('WebSocket 错误', true); };
      ws.onclose = function (ev) {
        var was = S.connected;
        S.connected = false; S.connecting = false; S.responding = false;
        stopMic(); flushPlayback();
        setConn('bad', '已断开');
        setOrb();
        if (was) log('连接关闭 (' + ev.code + ')');
        if (!opened) { S.connecting = false; reject(new Error('closed before ready')); }
      };
    });
  }
  function hangup() {
    stopMic();
    if (S.ws) { try { S.ws.close(); } catch (e) {} S.ws = null; }
    S.connected = false; S.responding = false;
    flushPlayback();
    setConn('', '未连接');
    setOrb();
    log('会话已结束');
  }

  function handleEvent(j) {
    var t = j.type;
    switch (t) {
      case 'session.created':
        sendSession();
        break;
      case 'session.updated':
        log('会话配置就绪 · 音色 ' + S.voice + ' · ' + (S.vad ? '免提 VAD' : '按住说话'));
        refreshAggregate();
        break;
      case 'input_audio_buffer.speech_started':
        if (S.responding || S.speaking) { flushPlayback(); log('插话打断', true); }
        S.userStartMs = j.audio_start_ms || 0;
        addUserClip();
        setOrb();
        break;
      case 'input_audio_buffer.speech_stopped':
        finishUserClip((j.audio_end_ms || 0) - (S.userStartMs || 0));
        break;
      case 'input_audio_buffer.committed':
        if (S.userClip) finishUserClip(null);
        log('音频已提交');
        break;
      case 'response.created':
        S.responding = true;
        addBotBubble();
        setOrb();
        log('开始生成回答');
        break;
      case 'response.audio_transcript.delta':
      case 'response.output_audio_transcript.delta':
        botDelta(j.delta || '');
        break;
      case 'response.audio.delta':
      case 'response.output_audio.delta':
        if (j.delta) playDelta(b64ToFloat(j.delta));
        break;
      case 'response.audio_transcript.done':
      case 'response.output_audio_transcript.done':
        if (S.turn && j.transcript) {
          S.turn.text = j.transcript;
          S.turn.el.querySelector('.txt').textContent = j.transcript;
        }
        break;
      case 'response.done':
        S.responding = false;
        var resp = j.response || {};
        var cw = resp.cwick || null;
        if (S.turn) S.turn.cw = cw;
        updatePerf(cw);
        finishBot(resp.status || 'completed');
        refreshAggregate();
        setOrb();
        log('回答完成 · ' + (resp.status || 'completed') +
          (cw ? ' · 首音频 ' + cw.first_audio_s.toFixed(2) + 's' : ''), true);
        break;
      case 'error':
        var em = (j.error && j.error.message) || '未知错误';
        log('错误：' + em, true);
        if (S.responding) { S.responding = false; finishBot('failed'); setOrb(); }
        break;
      default:
        break;
    }
  }

  /* ---------------------------------------------------------------- */
  /* text input                                                        */
  /* ---------------------------------------------------------------- */
  /* The speech model has no text-only turn: a text item is folded into the
   * instructions the next spoken turn runs under. So the text box sets a
   * teaching directive, then the user asks by voice. */
  async function sendText() {
    var inp = $('textInput');
    var text = (inp.value || '').trim();
    if (!text) return;
    try { await connect(); } catch (e) { alert('无法连接 /v1/realtime：' + e.message); return; }
    send({ type: 'conversation.item.create', item: { type: 'message', role: 'user',
      content: [{ type: 'input_text', text: text }] } });
    clearEmpty();
    var div = document.createElement('div');
    div.className = 'msg bot';
    div.innerHTML = '<span class="who">教学指令</span><div class="bubble">' + esc(text) +
      '<br><small style="color:var(--muted)">已交给模型 · 点击下方麦克风，直接说话提问。</small></div>';
    $('convo').appendChild(div); scrollConvo();
    inp.value = '';
    log('教学指令已设置 · 请说话提问');
    if (!S.micOn) await startMic();
  }

  /* ---------------------------------------------------------------- */
  /* storage                                                           */
  /* ---------------------------------------------------------------- */
  var STORE = { app: 'teachme', ok: null };
  function lsGet(k) { try { return (JSON.parse(localStorage.getItem(LS) || '{}') || {})[k]; } catch (e) { return null; } }
  function lsPut(k, v) { try { var j = JSON.parse(localStorage.getItem(LS) || '{}') || {}; j[k] = v; localStorage.setItem(LS, JSON.stringify(j)); } catch (e) {} }
  async function storeProbe() {
    try { var r = await fetch('/v1/store?app=' + STORE.app, { cache: 'no-store' });
      if (authRedirect(r)) return; STORE.ok = r.ok; if (r.ok) await r.json(); }
    catch (e) { STORE.ok = false; }
  }
  async function storePut(k, v) {
    lsPut(k, v);
    if (STORE.ok === false) return;
    try {
      var r = await fetch('/v1/apps/' + STORE.app + '/state', {
        method: 'PUT', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ key: k, value: v })
      });
      if (!r.ok) STORE.ok = false;
    } catch (e) { STORE.ok = false; }
  }
  function persist() {
    var slim = S.turns.slice(-40).map(function (t) {
      return { role: t.role, text: t.text, secs: t.secs, status: t.status, at: t.at,
        first: t.cw ? t.cw.first_audio_s : null, turn: t.cw ? t.cw.turn_s : null };
    });
    storePut('convo', slim);
  }
  async function loadHistory() {
    var h = null;
    if (STORE.ok !== false) {
      try {
        var r = await fetch('/v1/apps/' + STORE.app + '/state?key=convo', { cache: 'no-store' });
        if (r.ok) { var j = await r.json(); h = j && j.value; }
      } catch (e) {}
    }
    if (!h) h = lsGet('convo');
    if (!Array.isArray(h) || !h.length) return;
    clearEmpty();
    h.forEach(function (t) {
      var div = document.createElement('div');
      div.className = 'msg ' + (t.role === 'user' ? 'user' : 'bot');
      if (t.role === 'user') {
        var body = t.text ? esc(t.text) : ('🎤 语音 · ' + (t.secs ? t.secs.toFixed(1) : '?') + ' s');
        div.innerHTML = '<span class="who">你</span><div class="bubble">' + body + '</div>';
      } else {
        div.innerHTML = '<span class="who">智子 Sophon</span><div class="bubble">' + esc(t.text || '') + '</div>' +
          '<div class="meta"><span class="timing">' +
          (t.first != null ? '首音频 ' + t.first.toFixed(2) + 's' : '') +
          (t.status === 'cancelled' ? ' · 已被打断' : '') + '</span></div>';
      }
      $('convo').appendChild(div);
    });
    scrollConvo();
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
      var res = await fetch('/v1/share?app=teachme', { cache: 'no-store' });
      if (authRedirect(res)) return;
      if (!res.ok) throw new Error('HTTP ' + res.status);
      var j = await res.json();
      var url = j.app_url ||
        ((j.lan_urls && j.lan_urls.length) ? j.lan_urls[0] + '/apps/teachme/index.html' : '');
      if (!j.shareable || !url) {
        urlEl.textContent = location.origin + '/apps/teachme/index.html';
        note.className = 'share-note bad';
        note.textContent = (j.warning || '当前服务没有可分享的局域网地址。') + ' 用 --lan 启动后重试。';
        return;
      }
      urlEl.textContent = url;
      img.src = '/v1/share/qr?text=' + encodeURIComponent(url);
      img.hidden = false;
      note.textContent = '手机与这台 Mac 连同一个 Wi-Fi，扫码后即可独立使用；首次打开需要输入登录密码。';
    } catch (e) {
      urlEl.textContent = location.origin + '/apps/teachme/index.html';
      note.className = 'share-note bad';
      note.textContent = '无法读取 /v1/share：' + (e && e.message ? e.message : e);
    }
  }

  /* ---------------------------------------------------------------- */
  /* bind / boot                                                       */
  /* ---------------------------------------------------------------- */
  function bind() {
    $('mic').addEventListener('click', async function () {
      if (!S.connected) {
        try { await connect(); } catch (e) { alert('无法连接 /v1/realtime：' + e.message); return; }
        await startMic();
        return;
      }
      if (S.micOn) {
        stopMic();
        if (!S.vad) {
          finishUserClip(null);
          send({ type: 'input_audio_buffer.commit' });
          send({ type: 'response.create' });
        }
      } else {
        await startMic();
        if (!S.vad) addUserClip();
      }
    });
    $('hangup').addEventListener('click', hangup);
    $('interrupt').addEventListener('click', function () {
      flushPlayback();
      send({ type: 'response.cancel' });
      log('手动打断', true);
    });
    $('mode').addEventListener('click', function () {
      S.vad = !S.vad;
      $('mode').textContent = S.vad ? '免提 VAD' : '按住说话';
      if (S.connected) sendSession();
      setOrb();
      log('切换为 ' + (S.vad ? '免提 VAD' : '按住说话'));
    });
    $('voice').addEventListener('change', function () {
      S.voice = $('voice').value;
      if (S.connected) sendSession();
      log('音色切换为 ' + S.voice);
    });
    $('textSend').addEventListener('click', sendText);
    $('textInput').addEventListener('keydown', function (e) {
      if (e.key === 'Enter') { e.preventDefault(); sendText(); }
    });
    $('refreshStats').addEventListener('click', refreshAggregate);
    $('clearLog').addEventListener('click', function () { $('log').innerHTML = ''; });
    $('share').addEventListener('click', share);
    $('closeShare').addEventListener('click', function () {
      var d = $('shareDialog'); if (d.close) d.close(); else d.removeAttribute('open');
    });
    window.addEventListener('beforeunload', function () { if (S.ws) try { S.ws.close(); } catch (e) {} });
  }

  async function boot() {
    bind();
    setOrb();
    await storeProbe();
    await loadHistory();
    refreshAggregate();
    /* Warm the playback context on first gesture (autoplay policy). */
    document.addEventListener('pointerdown', function warm() {
      ensurePlayCtx();
      if (S.playCtx && S.playCtx.state === 'suspended') S.playCtx.resume();
      document.removeEventListener('pointerdown', warm);
    }, { once: true });
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
})();