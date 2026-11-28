(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const STORE_KEY = 'cwick.music.v2';
  const TEXT_MODEL_KEY = 'cwick.music.text-model';
  const MUSIC_ID = 'MiniMax-Music3';
  const PHASE_NAMES = {
    load: '加载模型', tokenize: '提示词', reference_encode: '参考音频编码',
    global_lm: '全局语言模型', depth: 'RVQ 深度解码', condition: '条件编码',
    flow: 'Flow 生成', vocoder: '声码器', wav: 'WAV 封装', done: '完成'
  };

  const IDEA_THEMES = [
    '雨夜末班车上的短暂相遇', '海边旧车站与未寄出的信', '凌晨便利店里的蓝色灯光',
    '沙漠公路尽头的第一场雪', '旧唱片店顶层的小型演唱会', '月球花园里盛开的电子花',
    '山城清晨雾里的早餐摊', '夏日屋顶上远去的纸飞机', '冬夜厨房里的一碗热汤',
    '废弃剧院重新亮起的舞台灯', '雨后城市屋顶的爵士即兴', '森林深处会发光的溪流',
    '给十年后自己的一段留言', '跨年夜倒计时前的最后一次犹豫', '老电影院散场后的长街',
    '一杯咖啡从热到凉的时间', '远方渔船带回的黎明', '地铁站里错过又重逢的人',
    '凌晨四点仍然亮着的书桌', '风吹过麦田时的故乡', '潜水艇窗外深海的鲸歌',
    '春天第一条解冻的小溪', '在城市天台种下一棵树', '一场只下在回忆里的雪'
  ];
  const encoder = new TextEncoder();
  const decoder = new TextDecoder();
  const byteLength = (value) => encoder.encode(value).length;
  const fitBytes = (value, limit) => {
    const bytes = encoder.encode(value);
    return bytes.length <= limit ? value : decoder.decode(bytes.slice(0, limit)).replace(/\uFFFD+$/, '');
  };

  const S = {
    textModel: '', textLoaded: false, musicLoaded: false,
    textFallback: '', busy: false, pollTimer: 0, pollFailures: 0, taskId: '',
    objectUrl: '', ideaSerial: 0, ideaSeconds: 0
  };

  function notice(text, bad = true) {
    $('notice').textContent = text || '';
    $('notice').style.color = bad ? 'var(--danger)' : 'var(--accent-2)';
  }

  function history() {
    try {
      const value = JSON.parse(localStorage.getItem(STORE_KEY) || '[]');
      return Array.isArray(value) ? value : [];
    } catch (_) {
      return [];
    }
  }

  function saveHistory(items) {
    localStorage.setItem(STORE_KEY, JSON.stringify(items.slice(0, 8)));
    renderHistory();
  }

  function remember(job, idea) {
    const items = history().filter((item) => item.taskId !== job.id);
    items.unshift({
      taskId: job.id,
      title: job.title || idea.title || '',
      prompt: job.style || idea.style || '',
      created: job.created_at || Math.floor(Date.now() / 1000),
      status: job.status || 'queued'
    });
    saveHistory(items);
  }

  function renderHistory() {
    const items = history();
    const list = $('history');
    list.replaceChildren();
    if (!items.length) {
      const li = document.createElement('li');
      li.className = 'muted';
      li.textContent = '暂无历史';
      list.appendChild(li);
      return;
    }
    for (const item of items) {
      const li = document.createElement('li');
      const button = document.createElement('button');
      button.type = 'button';
      button.disabled = S.busy;
      button.textContent = item.title || item.prompt || item.taskId;
      button.title = item.prompt || item.taskId;
      button.addEventListener('click', () => openTask(item.taskId));
      const state = document.createElement('span');
      state.className = 'mono muted';
      state.textContent = item.status === 'completed' ? '完成' : item.status;
      li.append(button, state);
      list.appendChild(li);
    }
  }

  async function api(path, options) {
    const response = await fetch(path, options);
    if (response.status === 401) {
      location.href = '/login';
      throw new Error('登录已过期');
    }
    let body = null;
    try { body = await response.json(); } catch (_) {}
    if (!response.ok) {
      const message = body?.msg || body?.error?.message || body?.detail || `HTTP ${response.status}`;
      const error = new Error(message);
      error.status = response.status;
      throw error;
    }
    return body;
  }

  function postJson(path, payload) {
    return api(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
  }

  /* Both states come from the live catalog: /v1/models publishes which model
   * the server actually has resident, and a runtime switch can move that flag
   * off index 0. Text entries have no `cwick.kind`; the pipeline models do. */
  async function readModels() {
    try {
      const j = await api('/v1/models', { cache: 'no-store' });
      const data = Array.isArray(j.data) ? j.data : [];
      let loadedText = null;
      let fallbackText = null;
      let music = null;
      for (const entry of data) {
        const c = entry.cwick || {};
        if (!c.kind && c.supported !== false) {
          if (!fallbackText) fallbackText = entry.id;
          if (c.loaded && !loadedText) loadedText = entry;
        }
        if ((c.kind === 'music' || entry.id === MUSIC_ID) && !music) music = entry;
      }
      S.textFallback = fallbackText || '';
      S.textLoaded = !!loadedText;
      S.musicLoaded = !!(music && music.cwick && music.cwick.loaded);
      if (loadedText && loadedText.id) {
        S.textModel = loadedText.id;
        localStorage.setItem(TEXT_MODEL_KEY, loadedText.id);
      } else {
        const saved = localStorage.getItem(TEXT_MODEL_KEY) || '';
        S.textModel = saved && data.some((e) => e.id === saved && !e.cwick?.kind)
          ? saved : (fallbackText || '');
      }
      $('textPill').textContent = S.textLoaded
        ? `文本模型已就绪 · ${S.textModel}` : '文本模型未加载';
      $('textPill').className = `pill${S.textLoaded ? ' ready' : ''}`;
      $('musicPill').textContent = S.musicLoaded ? '音乐模型已就绪' : '音乐模型按需加载';
      $('musicPill').className = `pill${S.musicLoaded ? ' ready' : ''}`;
      $('switchMusic').disabled = S.busy || S.musicLoaded;
      $('ideaBtn').disabled = S.busy;
      return true;
    } catch (error) {
      $('textPill').textContent = $('musicPill').textContent = '模型状态不可用';
      notice(error.message);
      return false;
    }
  }

  async function ensureModel(which) {
    await readModels();
    if (which === 'music') {
      if (S.musicLoaded) return;
      await postJson('/v1/models/switch', { model: MUSIC_ID });
    } else {
      if (S.textLoaded) return;
      if (!S.textModel) throw new Error('目录里没有可用的文本模型');
      await postJson('/v1/models/switch', { model: S.textModel });
    }
    await readModels();
  }

  function setBusy(value) {
    S.busy = value;
    $('ideaBtn').disabled = value;
    $('submit').disabled = value;
    $('switchMusic').disabled = value || S.musicLoaded;
    for (const input of document.querySelectorAll('#musicForm input, #musicForm textarea'))
      input.disabled = value;
    renderHistory();
  }

  /* The checkpoint takes a three-section Structured Caption. For pure music,
   * Vocal Details explicitly forbids singers, lyrics and vocal samples. */
  const CAPTION_SHAPE = [
    'Global Metadata',
    'Basic Attributes: bpm is <number>. key is <note>, and scale is <major or minor>. <genre / subgenre>.',
    'Global Emotional Progression: <how the mood moves across the track>.',
    'Application Scenarios & Imagery: <where this instrumental belongs>.',
    'Sonics & Production Profile: <mix, width, low end, dynamics>.',
    'Vocal Details',
    'Vocals: None. Pure instrumental; no singer, no lyrics, no sung text, no vocal samples.',
    'Instrumental Focus: <lead instrument, melody, timbre and register>.',
    'Vocal FX: None. Do not add vocal chops, harmonies, whispers or processed vocals.',
    'Arrangement',
    'Instrument Lifecycle Description (Primary/Secondary Layering): <primary and secondary instruments and when they enter>.',
    'Groove & Foundation Progression: <drums, percussion, bass>.',
    'Embellishments, Textures & Spatial FX: <short fills, clean textures, space; avoid continuous noise washes>.'
  ].join('\n');

  function buildCaptionRequest(topic, seconds, correction = '') {
    const short = seconds <= 15;
    const medium = seconds > 15 && seconds <= 60;
    const wordCount = short ? '160-260' : medium ? '220-360' : '250-450';
    const density = short
      ? 'This is a short track. Use at most one lead instrument plus a minimal bass/percussion foundation. '
        + 'Do not layer rain, wind, crowd, noise beds, vocal chops, or multiple competing melodies. '
        + 'Keep the lead melody clear and the arrangement sparse enough for a short duration.'
      : medium
        ? 'Use a focused arrangement with one lead instrument and a small supporting ensemble. Avoid dense noise beds and competing melodies.'
        : 'Use a full but coherent arrangement with clear primary and secondary layers.';
    const timeline = short
      ? `Use a simple three-part timeline: 0-2s intro, 2-${Math.max(3, seconds - 2)}s main body, final 2s outro. `
        + 'Keep at least one pitched instrument active until the final 0.5 seconds. No silent gaps, no sudden late re-entry, '
        + 'and no more than three instrument entrances. Keep overall loudness stable; avoid dramatic quiet-loud swings.'
      : 'Arrangement must be a section-by-section timeline with instrument entrances, exits and energy changes.';
    return {
      system: 'You are a professional instrumental music producer. Output only an English Structured Caption, '
        + 'with no JSON, no markdown, no bullets, no title, no lyrics, and no commentary. Use exactly these '
        + 'three top-level headings in this order and write ' + wordCount + ' English words:\n'
        + CAPTION_SHAPE + '\n'
        + `The complete instrumental track must fit ${seconds} seconds. Keep the stated BPM stable, make the `
        + 'arrangement fit the full duration, and reserve the final section for a clear Outro. '
        + 'Vocals must be None: no singer, no lyrics, no sung text, no vocal samples, no vocal chops, no '
        + 'harmonies, and no whispers. Do not request a lead vocal or an instrumental intro followed by vocals. '
        + 'Describe a memorable instrumental melody, harmonic progression and rhythm section. Keep the lead '
        + 'instrument clear: no continuous white-noise washes, no dense effects that mask the melody, and no '
        + 'unstructured improvisation. Never use the words silence or silent. Keep audible music through the final 0.5 seconds. '
        + density + ' ' + timeline,
      user: `Theme: ${topic}` + (correction ? `\nThe previous caption was rejected: ${correction} Return a corrected complete caption.` : '')
    };
  }

  function parseIdea(raw) {
    const text = (raw || '').trim();
    if (!text) throw new Error('文本模型没有返回内容');
    const grab = (name) => {
      const marker = `"${name}"`;
      let at = text.indexOf(marker);
      if (at < 0) return '';
      at += marker.length;
      while (/\s/.test(text[at] || '')) at++;
      if (text[at] !== ':') return '';
      at++;
      while (/\s/.test(text[at] || '')) at++;
      if (text[at] !== '"') return '';
      at++;
      let value = '';
      let escaped = false;
      while (at < text.length) {
        const ch = text[at++];
        if (escaped) {
          if (ch === 'n') value += '\n';
          else if (ch === 'r') value += '\r';
          else if (ch === 't') value += '\t';
          else if (ch === 'u' && at + 4 <= text.length) {
            const hex = text.slice(at, at + 4);
            if (/^[0-9a-f]{4}$/i.test(hex)) { value += String.fromCharCode(parseInt(hex, 16)); at += 4; }
          } else value += ch;
          escaped = false;
        } else if (ch === '\\') escaped = true;
        else if (ch === '"') return value;
        else value += ch;
      }
      return '';
    };
    const addSections = (style, source) => {
      let out = style;
      for (const heading of ['Global Metadata', 'Vocal Details', 'Arrangement']) {
        const part = source ? source[heading] : grab(heading);
        if (typeof part === 'string' && part.trim() && !out.includes(heading))
          out += (out ? '\n' : '') + heading + '\n' + part.trim();
      }
      return out;
    };
    let obj = null;
    const start = text.indexOf('{');
    const end = text.lastIndexOf('}');
    if (start >= 0 && end > start) {
      try { obj = JSON.parse(text.slice(start, end + 1)); } catch (_) {}
    }
    if (obj && typeof obj === 'object') {
      let style = typeof obj.style === 'string' ? obj.style : (typeof obj.prompt === 'string' ? obj.prompt : '');
      if (obj.style && typeof obj.style === 'object')
        style = Object.entries(obj.style).map(([key, value]) => key + '\n' + value).join('\n');
      style = addSections(style, obj);
      const title = typeof obj.title === 'string' ? obj.title : '';
      if (!style) throw new Error('文本模型返回的 JSON 里没有描述');
      return { style, title };
    }
    const style = addSections(grab('style') || grab('prompt'), null);
    if (!style) throw new Error('文本模型没有按 JSON 格式返回描述');
    return { style, title: grab('title') };
  }

  function validateCaption(raw, seconds) {
    let style = (raw || '').trim();
    if (style.startsWith('{')) style = parseIdea(style).style;
    style = style.replace(/```(?:text|json)?/gi, '').replace(/^#{1,6}\s+/gm, '').trim();
    if (!/^Global Metadata$/im.test(style) && /^Basic Attributes:/im.test(style))
      style = 'Global Metadata\n' + style;
    if (!/^Vocal Details$/im.test(style) && /^Vocals:/im.test(style))
      style = style.replace(/^Vocals:/im, 'Vocal Details\nVocals:');
    if (!/^Arrangement$/im.test(style) && /^Instrument Lifecycle Description/im.test(style))
      style = style.replace(/^Instrument Lifecycle Description/im, 'Arrangement\nInstrument Lifecycle Description');
    for (const heading of ['Global Metadata', 'Vocal Details', 'Arrangement']) {
      if (!new RegExp('^' + heading + '$', 'im').test(style))
        throw new Error('描述缺少官方必需段落：' + heading);
    }
    if (!/^Vocals:\s*None\.?/im.test(style))
      throw new Error('Vocal Details 必须明确 Vocals: None');
    if (/\b(silence|silent)\b/i.test(style))
      throw new Error('短曲描述不能要求静音，请保持声音延续到结尾');
    if (!/^Instrumental Focus:/im.test(style))
      throw new Error('Vocal Details 必须明确主要乐器和旋律');
    if (style.length < 900 || byteLength(style) > 4000)
      throw new Error('描述长度不符合官方 Structured Caption 要求（900–4000 字节）');
    const arrangement = style.slice(style.search(/^Arrangement$/im));
    for (const match of arrangement.matchAll(/\b(\d+):(\d{2})\b/g)) {
      if (Number(match[1]) * 60 + Number(match[2]) > seconds)
        throw new Error(`编排时间线超过 ${seconds} 秒，请压缩到请求时长`);
    }
    for (const match of arrangement.matchAll(/\b(\d+)\s*(?:seconds|sec)\b/gi)) {
      if (Number(match[1]) > seconds)
        throw new Error(`编排时间线超过 ${seconds} 秒，请压缩到请求时长`);
    }
    return style;
  }

  async function generateIdea() {
    if (S.busy) return null;
    const serial = ++S.ideaSerial;
    setBusy(true);
    $('taskState').textContent = '正在调用文本模型写纯音乐描述…';
    $('phasePill').textContent = '描述生成中';
    notice('');
    try {
      await ensureModel('text');
      const seconds = Math.min(240, Math.max(10, Number($('duration').value) || 60));
      const topic = $('theme').value.trim() || IDEA_THEMES[Math.floor(Math.random() * IDEA_THEMES.length)];
      const title = topic.slice(0, 40);
      $('title').value = fitBytes(title, 80);
      let style = '';
      let captionError = null;
      for (let attempt = 0; attempt < 2; attempt++) {
        const req = buildCaptionRequest(topic, seconds, captionError ? captionError.message : '');
        const res = await postJson('/v1/chat/completions', {
          model: S.textModel,
          messages: [
            { role: 'system', content: req.system },
            { role: 'user', content: req.user }
          ],
          max_tokens: 2200,
          temperature: 0.7,
          top_p: 0.95,
          reasoning_effort: 'none',
          stream: false
        });
        if (serial !== S.ideaSerial) return null;
        const msg = res.choices?.[0]?.message || {};
        try {
          style = validateCaption(msg.content || msg.reasoning_content, seconds);
          break;
        } catch (error) {
          captionError = error;
        }
      }
      if (!style) throw new Error('文本模型连续两次未返回完整纯音乐 Structured Caption：' + captionError.message);
      $('prompt').value = fitBytes(style, 4000);
      S.ideaSeconds = seconds;
      $('ideaMeta').hidden = false;
      $('ideaMeta').textContent = `由 ${S.textModel} 生成 · ${seconds} 秒 · 纯音乐 · 描述 ${byteLength(style)} 字节`;
      notice('纯音乐描述已生成，可以直接修改后生成', false);
      $('taskState').textContent = '灵感就绪，点击「一键生成纯音乐」';
      return { title, style };
    } catch (error) {
      if (serial !== S.ideaSerial) return null;
      $('taskState').textContent = '灵感生成失败';
      notice(error.message);
      throw error;
    } finally {
      if (serial === S.ideaSerial) {
        setBusy(false);
        readModels();
      }
    }
  }

  async function submit(event) {
    event.preventDefault();
    if (S.busy) return;
    notice('');
    let style = $('prompt').value.trim();
    let submitted = false;
    try {
      if (!style) {
        const idea = await generateIdea();
        if (!idea) return;
        style = $('prompt').value.trim();
      }
      if (!style) throw new Error('请先生成或填写纯音乐描述');
      const seconds = Math.min(240, Math.max(10, Number($('duration').value) || 60));
      if (S.ideaSeconds && S.ideaSeconds !== seconds)
        throw new Error(`描述按 ${S.ideaSeconds} 秒生成，当前时长为 ${seconds} 秒；请重新生成灵感`);
      style = fitBytes(style, 4000);
      if (!/^Vocals:\s*None\.?/im.test(style))
        throw new Error('纯音乐描述必须明确 Vocals: None，禁止主唱和歌词');
      const seedText = $('seed').value.trim();
      const seed = seedText ? Number(seedText) : Math.floor(Math.random() * 9007199254740991);
      if (!Number.isSafeInteger(seed) || seed < 0 || seed > 9007199254740991)
        throw new Error('Seed 必须是 0 到 9007199254740991 之间的整数');
      const payload = {
        model: MUSIC_ID,
        customMode: true,
        task: 'generate',
        instrumental: true,
        style,
        title: fitBytes($('title').value.trim() || `灵感 ${new Date().toISOString().slice(0, 10)}`, 80),
        lyrics: seconds <= 15 ? '[Intro]\n[Instrumental]\n[Outro]' : '[Instrumental]',
        duration: seconds,
        steps: Math.min(30, Math.max(1, Number($('steps').value) || 30)),
        cfg_scale: 1.7,
        seed
      };
      setBusy(true);
      await ensureModel('music');
      $('taskState').textContent = '正在提交生成任务…';
      $('phasePill').textContent = '提交中';
      const accepted = await postJson('/v1/generate', payload);
      const id = accepted.data?.taskId || accepted.data?.id;
      if (!id) throw new Error('服务端没有返回 taskId');
      submitted = true;
      S.taskId = id;
      remember({ id, ...accepted.data, ...payload }, { title: payload.title, style: payload.style });
      notice('纯音乐任务已提交，正在实时轮询进度', false);
      openTask(id);
    } catch (error) {
      $('taskState').textContent = '提交失败';
      notice(error.message);
    } finally {
      if (!submitted) setBusy(false);
      readModels();
    }
  }

  function renderJob(job) {
    S.taskId = job.id;
    $('taskState').textContent = `生成任务 · ${job.id}`;
    const percent = Math.round((job.progress || 0) * 100);
    $('phasePill').textContent = `${PHASE_NAMES[job.phase] || job.phase || job.status} ${percent}%`;
    $('phasePill').className = `pill${job.status === 'completed' ? ' ready' : job.status === 'failed' ? ' busy' : ''}`;
    $('progressBar').style.width = `${percent}%`;
    $('progressBar').parentElement.setAttribute('aria-valuenow', String(percent));
    $('progressText').textContent = `${percent}%`;
    $('stepText').textContent = job.steps_total ? `flow ${job.steps_done}/${job.steps_total}` : '—';
    $('ideaMeta').hidden = false;
    $('ideaMeta').textContent = `本次请求：纯音乐 · 描述 ${byteLength(job.prompt || '')} 字节${job.timing?.prompt_tokens ? ` · prompt ${job.timing.prompt_tokens} tokens` : ''}${job.timing?.ended_early ? ` · 提前 EOS（${job.timing.frames}/${job.timing.requested_frames} 帧）` : ''}`;

    const stages = $('stages');
    stages.replaceChildren();
    for (const stage of job.stage_history || []) {
      const li = document.createElement('li');
      const name = document.createElement('span');
      name.textContent = PHASE_NAMES[stage.phase] || stage.phase;
      const value = document.createElement('span');
      value.className = 'stage-time';
      const stagePercent = job.status === 'completed' ? 100 : (stage.total > 0
        ? Math.max(0, Math.min(100, Math.round(stage.done / stage.total * 100)))
        : (stage.phase === job.phase ? percent : 100));
      const amount = stage.duration_ms
        ? `${(stage.duration_ms / 1000).toFixed(2)} s`
        : `${stage.done}/${stage.total}`;
      const frameNote = stage.phase === 'depth' && stage.total > 0 ? ` · ${stage.done}/${stage.total} 帧` : '';
      value.textContent = job.status === 'completed' && stage.done < stage.total
        ? `已结束 · ${amount}${frameNote}` : `${stagePercent}% · ${amount}${frameNote}`;
      li.append(name, value);
      li.className = stage.phase === job.phase ? 'active' : 'done';
      stages.appendChild(li);
    }
    if (!(job.stage_history || []).length) {
      const li = document.createElement('li');
      li.className = 'muted';
      li.textContent = job.phase || job.status;
      stages.appendChild(li);
    }

    const timing = $('timing');
    timing.replaceChildren();
    if (job.timing) {
      for (const [key, label] of Object.entries({
        total_ms: '总耗时', load_ms: '加载', tokenize_ms: '分词', reference_encode_ms: '参考编码',
        global_lm_ms: 'Global LM', depth_ms: 'Depth', condition_ms: '条件', flow_ms: 'Flow',
        vocoder_ms: '声码器', wav_ms: 'WAV'
      })) {
        if (!(key in job.timing)) continue;
        const span = document.createElement('span');
        const name = document.createElement('i');
        name.style.fontStyle = 'normal';
        name.textContent = label;
        const value = document.createElement('b');
        value.textContent = `${(job.timing[key] / 1000).toFixed(2)} s`;
        span.append(name, value);
        timing.appendChild(span);
      }
    } else {
      const span = document.createElement('span');
      span.className = 'muted';
      span.textContent = '完成任务后显示各阶段毫秒数';
      timing.appendChild(span);
    }

    const items = history();
    const current = items.find((item) => item.taskId === job.id);
    if (current && current.status !== job.status) {
      current.status = job.status;
      saveHistory(items);
    }
  }

  async function loadOutput(job) {
    const response = await fetch(job.content_url || `/v1/generate/${job.id}/content`);
    if (response.status === 401) { location.href = '/login'; return; }
    if (!response.ok) throw new Error(`音频下载失败：HTTP ${response.status}`);
    const blob = await response.blob();
    if (S.objectUrl) URL.revokeObjectURL(S.objectUrl);
    S.objectUrl = URL.createObjectURL(blob);
    $('player').src = S.objectUrl;
    $('player').hidden = false;
    $('emptyPlayer').hidden = true;
    $('download').href = S.objectUrl;
    $('download').hidden = false;
    $('download').download = `${(job.title || job.id).replace(/[\\/:*?"<>|]/g, '_')}.wav`;
  }

  async function openTask(id) {
    clearTimeout(S.pollTimer);
    S.taskId = id;
    $('taskState').textContent = `正在查询 ${id}…`;
    try {
      const job = await api(`/v1/generate/${encodeURIComponent(id)}`);
      S.pollFailures = 0;
      renderJob(job);
      if (job.status === 'completed') {
        await loadOutput(job);
        setBusy(false);
        readModels();
        notice('音乐已生成，可在线播放或下载 WAV', false);
        return;
      }
      if (job.status === 'failed' || job.status === 'cancelled') {
        setBusy(false);
        readModels();
        notice(job.error?.message || `任务${job.status}`);
        return;
      }
      const slowPhase = ['global_lm', 'depth', 'flow', 'vocoder'].includes(job.phase);
      S.pollTimer = setTimeout(() => openTask(id), slowPhase ? 2000 : 1000);
    } catch (error) {
      if (error.status === 404) {
        setBusy(false);
        readModels();
        notice('任务已不存在（服务重启后内存任务会清空）');
        return;
      }
      S.pollFailures++;
      const delay = Math.min(30000, 1000 * (2 ** Math.min(S.pollFailures - 1, 5)));
      notice(`进度查询失败，${Math.round(delay / 1000)} 秒后重试：${error.message}`);
      S.pollTimer = setTimeout(() => openTask(id), delay);
    }
  }

  $('ideaBtn').addEventListener('click', () => {
    generateIdea().catch(() => {});
  });
  $('musicForm').addEventListener('submit', submit);
  $('switchMusic').addEventListener('click', async () => {
    try {
      setBusy(true);
      await ensureModel('music');
      notice('音乐模型已就绪，可以一键生成', false);
    } catch (error) {
      notice(error.message);
    } finally {
      setBusy(false);
      readModels();
    }
  });
  $('clearHistory').addEventListener('click', () => saveHistory([]));
  window.addEventListener('beforeunload', () => {
    if (S.objectUrl) URL.revokeObjectURL(S.objectUrl);
  });

  setBusy(false);
  readModels();
})();
