/* cwick web console.
 *
 * Five views over the same server:
 *   chat     — OpenAI-compatible /v1/chat/completions or /v1/responses, or
 *              Anthropic's /v1/messages (the composer picks one), streamed
 *              over SSE, with the checkpoint's reasoning shown separately.
 *              All three reach the same engine; the page translates each
 *              envelope into the chat shape so every widget below it has one
 *              input.
 *   apps     — a manifest-driven H5 marketplace. apps/manifest.json lists
 *              the bundles compiled into the binary; opening one points a
 *              same-origin iframe at /apps/<id>/, so the app owns its own
 *              DOM/CSS/JS and reaches /v1/systemone without CORS.
 *   models   — the local model library: what is on disk under ./models, live
 *              download progress with pause/resume, and read-only catalog
 *              search. Downloads are executed by the server, never the page.
 *   logs     — tails /v1/logs/stream: the process's own stdout/stderr, which
 *              the server tees into a bounded ring before printing it.
 *   settings — the console and settings in one place: live /v1/metrics with
 *              four-minute trend charts, full model_info labels, read-only
 *              /v1/config, generation preferences, and /v1/models + POST
 *              /v1/models/switch to replace the resident local checkpoint.
 *
 * Every request in this file is same-origin. There is no configurable external
 * OpenAI endpoint; the API keys the settings view can mint are credentials for
 * other clients to call *this* server, not an address this page talks to.
 * There is no build step and no dependency: this file is served from the
 * binary itself, so opening http://127.0.0.1:PORT/ is all it takes.
 */
(function () {
  'use strict';

  var $ = function (id) { return document.getElementById(id); };
  var LS_SESSIONS = 'cwick.sessions.v1';
  var LS_PREFS = 'cwick.prefs.v1';
  var IMAGE_DB = 'cwick.images.v1';
  var IMAGE_STORE = 'images';
  var MAX_IMAGE_SIDE = 1024;
  var MAX_DRAFT_IMAGES = 8;
  var MAX_IMAGE_BYTES = 64 * 1024 * 1024;
  var MAX_ROUNDS = 4;                 /* tool-call round trips per turn */
  var EMPTY_HTML = '';                /* captured from the served document */

  var state = {
    modelId: '',
    sessions: [],
    currentId: null,
    busy: false,
    sending: false,
    imageBusy: false,
    draftImages: [],
    abort: null,
    prefs: { api: 'chat', effort: 'extreme', maxTokens: 2048, stream: true, tools: false, lang: 'zh' },
    lastTokS: 0,
    view: 'chat',
    liveTimer: 0,
    railTimer: 0,
    streaming: null
  };

  /* ------------------------------------------------------------------ */
  /* small utilities                                                     */
  /* ------------------------------------------------------------------ */
  function esc(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }
  function fmtBytes(n) {
    if (!n || n < 0) return '—';
    var u = ['B', 'KiB', 'MiB', 'GiB', 'TiB'], i = 0, v = n;
    while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; }
    return (v >= 100 || i === 0 ? v.toFixed(0) : v.toFixed(1)) + ' ' + u[i];
  }
  function fmtNum(n) {
    if (n == null || !isFinite(n)) return '—';
    if (n >= 1e9) return (n / 1e9).toFixed(2) + 'B';
    if (n >= 1e6) return (n / 1e6).toFixed(2) + 'M';
    if (n >= 1e4) return (n / 1e3).toFixed(1) + 'K';
    return String(Math.round(n));
  }
  function fmtSec(s) {
    if (s == null || !isFinite(s) || s <= 0) return '—';
    if (s < 1) return Math.round(s * 1000) + ' ms';
    if (s < 60) return s.toFixed(2) + ' s';
    if (s < 3600) return Math.floor(s / 60) + 'm ' + Math.round(s % 60) + 's';
    return Math.floor(s / 3600) + 'h ' + Math.round((s % 3600) / 60) + 'm';
  }
  function nowMs() { return (window.performance && performance.now) ? performance.now() : Date.now(); }
  function uid() { return Date.now().toString(36) + Math.random().toString(36).slice(2, 7); }

  /* ------------------------------------------------------------------ */
  /* local images: blobs in IndexedDB, references in session JSON         */
  /* ------------------------------------------------------------------ */
  var imageDbPromise = null;
  var imageCache = new Map();
  var imageLoads = new Map();

  function imageRef(image) {
    if (!image || typeof image.id !== 'string' || !image.id) return null;
    return {
      id: image.id,
      name: String(image.name || 'image'),
      mime: String(image.mime || 'image/jpeg'),
      width: Number(image.width) || 0,
      height: Number(image.height) || 0,
      size: Number(image.size) || 0,
      createdAt: Number(image.createdAt) || 0
    };
  }
  function storedSession(s) {
    var out = {};
    Object.keys(s).forEach(function (k) { if (k !== 'messages') out[k] = s[k]; });
    out.messages = (s.messages || []).map(function (m) {
      var copy = {};
      Object.keys(m).forEach(function (k) { if (k !== 'images') copy[k] = m[k]; });
      if (Array.isArray(m.images)) {
        copy.images = m.images.map(imageRef).filter(Boolean);
        if (!copy.images.length) delete copy.images;
      }
      return copy;
    });
    return out;
  }
  function sessionImages(s) {
    var refs = [];
    (s.messages || []).forEach(function (m) {
      (m.images || []).forEach(function (image) {
        var ref = imageRef(image);
        if (ref) refs.push(ref);
      });
    });
    return refs;
  }
  function referencedImageIds(extra) {
    var ids = new Set();
    state.sessions.forEach(function (s) {
      sessionImages(s).forEach(function (image) { ids.add(image.id); });
    });
    (extra || []).forEach(function (image) {
      var ref = imageRef(image);
      if (ref) ids.add(ref.id);
    });
    return ids;
  }
  function releaseImage(id) {
    var cached = imageCache.get(id);
    if (cached && cached.url && cached.url !== cached.dataUrl) URL.revokeObjectURL(cached.url);
    imageCache.delete(id);
    imageLoads.delete(id);
  }
  function openImageDb() {
    if (!window.indexedDB) return Promise.reject(new Error(t('chat.imageFailed') + 'IndexedDB'));
    if (!imageDbPromise) {
      imageDbPromise = new Promise(function (resolve, reject) {
        var req = window.indexedDB.open(IMAGE_DB, 1);
        req.onupgradeneeded = function () {
          req.result.createObjectStore(IMAGE_STORE, { keyPath: 'id' });
        };
        req.onsuccess = function () { resolve(req.result); };
        req.onerror = function () { reject(req.error || new Error('IndexedDB open failed')); };
      });
    }
    return imageDbPromise;
  }
  function imageDbGet(id) {
    return openImageDb().then(function (db) {
      return new Promise(function (resolve, reject) {
        var req = db.transaction(IMAGE_STORE, 'readonly').objectStore(IMAGE_STORE).get(id);
        req.onsuccess = function () { resolve(req.result); };
        req.onerror = function () { reject(req.error); };
      });
    });
  }
  function imageDbAll() {
    return openImageDb().then(function (db) {
      return new Promise(function (resolve, reject) {
        var req = db.transaction(IMAGE_STORE, 'readonly').objectStore(IMAGE_STORE).getAll();
        req.onsuccess = function () { resolve(req.result || []); };
        req.onerror = function () { reject(req.error); };
      });
    });
  }
  function imageDbWrite(fn) {
    return openImageDb().then(function (db) {
      return new Promise(function (resolve, reject) {
        var tx = db.transaction(IMAGE_STORE, 'readwrite');
        tx.oncomplete = function () { resolve(); };
        tx.onerror = function () { reject(tx.error || new Error('IndexedDB write failed')); };
        tx.onabort = function () { reject(tx.error || new Error('IndexedDB write aborted')); };
        try { fn(tx.objectStore(IMAGE_STORE)); } catch (e) { reject(e); }
      });
    });
  }
  function discardImages(images) {
    var candidates = new Set();
    (images || []).forEach(function (image) {
      var ref = imageRef(image);
      if (ref) candidates.add(ref.id);
    });
    if (!candidates.size) return Promise.resolve();
    var keep = referencedImageIds(state.draftImages);
    var ids = [];
    candidates.forEach(function (id) { if (!keep.has(id)) ids.push(id); });
    if (!ids.length) return Promise.resolve();
    return imageDbWrite(function (store) {
      ids.forEach(function (id) { store.delete(id); });
    }).then(function () {
      ids.forEach(releaseImage);
    }).catch(function () { /* localStorage deletion must still succeed */ });
  }
  function pruneOrphanImages() {
    return imageDbAll().then(function (records) {
      var keep = referencedImageIds(state.draftImages), orphans = [];
      records.forEach(function (record) { if (!keep.has(record.id)) orphans.push(record); });
      return discardImages(orphans);
    }).catch(function () { /* storage can be unavailable in private browsing */ });
  }
  function ensureImageCapacity(bytes) {
    return imageDbAll().then(function (records) {
      var keep = referencedImageIds(state.draftImages), orphans = [], used = 0;
      records.forEach(function (record) {
        if (!keep.has(record.id)) orphans.push(record);
        else used += Number(record.size) || (record.blob && record.blob.size) || 0;
      });
      if (used + bytes > MAX_IMAGE_BYTES) {
        var err = new Error(t('chat.imageFull'));
        err.code = 'image-full';
        throw err;
      }
      if (!orphans.length) return;
      return imageDbWrite(function (store) {
        orphans.forEach(function (record) { store.delete(record.id); });
      }).then(function () { orphans.forEach(function (record) { releaseImage(record.id); }); });
    });
  }
  function fileDataUrl(file) {
    return new Promise(function (resolve, reject) {
      var reader = new FileReader();
      reader.onload = function () { resolve(reader.result); };
      reader.onerror = function () { reject(new Error(t('chat.imageFailed') + file.name)); };
      reader.readAsDataURL(file);
    });
  }
  function blobDataUrl(blob) {
    return new Promise(function (resolve, reject) {
      var reader = new FileReader();
      reader.onload = function () { resolve(reader.result); };
      reader.onerror = function () { reject(reader.error); };
      reader.readAsDataURL(blob);
    });
  }
  /* Match the image studio's reference-image path: decode locally, fit within
     1024px, and encode JPEG so the shared 8 MiB request ceiling stays useful. */
  function shrinkImageFile(file) {
    return fileDataUrl(file).then(function (dataUrl) {
      return new Promise(function (resolve, reject) {
        var img = new Image();
        img.onload = function () {
          try {
            var w = img.naturalWidth || img.width, h = img.naturalHeight || img.height;
            var scale = Math.min(1, MAX_IMAGE_SIDE / Math.max(w, h));
            var cw = Math.max(1, Math.round(w * scale)), ch = Math.max(1, Math.round(h * scale));
            var canvas = document.createElement('canvas');
            canvas.width = cw; canvas.height = ch;
            var ctx = canvas.getContext('2d');
            ctx.fillStyle = '#fff';
            ctx.fillRect(0, 0, cw, ch);
            ctx.drawImage(img, 0, 0, cw, ch);
            canvas.toBlob(function (blob) {
              if (blob) resolve({ blob: blob, width: cw, height: ch });
              else reject(new Error(t('chat.imageFailed') + file.name));
            }, 'image/jpeg', 0.82);
          } catch (e) { reject(e); }
        };
        img.onerror = function () { reject(new Error(t('chat.imageFailed') + file.name)); };
        img.src = dataUrl;
      });
    });
  }
  function cacheImage(record) {
    return blobDataUrl(record.blob).then(function (dataUrl) {
      var url = URL.createObjectURL ? URL.createObjectURL(record.blob) : dataUrl;
      imageCache.set(record.id, {
        blob: record.blob, dataUrl: dataUrl, url: url,
        name: record.name, mime: record.mime, size: record.size
      });
      return imageCache.get(record.id);
    });
  }
  function getImage(ref) {
    var cached = imageCache.get(ref.id);
    if (cached) return Promise.resolve(cached);
    if (imageLoads.has(ref.id)) return imageLoads.get(ref.id);
    var load = imageDbGet(ref.id).then(function (record) {
      if (!record || !record.blob) throw new Error(t('chat.imageMissing'));
      return cacheImage(record);
    });
    imageLoads.set(ref.id, load);
    load.then(function () { imageLoads.delete(ref.id); },
              function () { imageLoads.delete(ref.id); });
    return load;
  }
  function hydrateImages(images) {
    var unique = new Map();
    (images || []).forEach(function (image) {
      var ref = imageRef(image);
      if (ref) unique.set(ref.id, ref);
    });
    return Promise.all(Array.from(unique.values()).map(getImage));
  }
  function hydrateSessionImages(session) {
    return hydrateImages(sessionImages(session));
  }
  function imagePayload(ref) {
    var cached = imageCache.get(ref.id);
    if (!cached || !cached.dataUrl) throw new Error(t('chat.imageMissing'));
    return {
      url: cached.dataUrl,
      base64: String(cached.dataUrl).split(',')[1] || '',
      mime: ref.mime || cached.mime || 'image/jpeg'
    };
  }
  function imageFigureHtml(image, draft) {
    var ref = imageRef(image);
    var meta = ref.width && ref.height ? ref.width + '×' + ref.height : fmtBytes(ref.size);
    return '<figure class="msg-image loading" data-image-id="' + esc(ref.id) + '">' +
      '<img hidden alt="' + esc(ref.name) + '">' +
      (draft ? '<button type="button" class="image-remove" data-remove-image="' + esc(ref.id) +
              '" aria-label="' + esc(t('chat.removeImage')) + '" title="' +
              esc(t('chat.removeImage')) + '">×</button>' :
       '<button type="button" class="image-remove" data-remove-history-image="' + esc(ref.id) +
              '" aria-label="' + esc(t('chat.removeImage')) + '" title="' +
              esc(t('chat.removeImage')) + '">×</button>') +
      '<figcaption>' + esc(ref.name) + ' · ' + esc(meta) + '</figcaption></figure>';
  }
  function paintImageFigures(root, images) {
    if (!root) return;
    root.querySelectorAll('.msg-image[data-image-id]').forEach(function (figure) {
      var ref = imageRef(images.filter(function (image) {
        return image && image.id === figure.getAttribute('data-image-id');
      })[0]);
      if (!ref) return;
      getImage(ref).then(function (cached) {
        var img = figure.querySelector('img');
        img.src = cached.url;
        img.hidden = false;
        figure.classList.remove('loading');
      }).catch(function () {
        figure.classList.remove('loading');
        var caption = figure.querySelector('figcaption');
        if (caption) caption.textContent = t('chat.imageMissing');
      });
    });
  }
  function renderDraftImages() {
    var box = $('draftImages');
    if (!box) return;
    box.hidden = !state.draftImages.length;
    box.innerHTML = state.draftImages.map(function (image) { return imageFigureHtml(image, true); }).join('');
    box.querySelectorAll('.image-remove').forEach(function (button) {
      button.disabled = state.busy || state.sending || state.imageBusy;
    });
    paintImageFigures(box, state.draftImages);
  }
  function removeDraftImage(id) {
    var removed = state.draftImages.filter(function (image) { return image.id === id; });
    state.draftImages = state.draftImages.filter(function (image) { return image.id !== id; });
    renderDraftImages();
    discardImages(removed);
  }
  function setHistoryImageControls() {
    var box = $('messages');
    if (!box) return;
    box.querySelectorAll('[data-remove-history-image]').forEach(function (button) {
      button.disabled = state.busy || state.sending;
    });
  }
  function removeHistoryImage(id) {
    if (state.busy || state.sending) return;
    var s = current();
    if (!s) return;
    for (var i = 0; i < s.messages.length; i++) {
      var m = s.messages[i];
      if (!Array.isArray(m.images)) continue;
      var removed = m.images.filter(function (image) { return image.id === id; });
      if (!removed.length) continue;
      m.images = m.images.filter(function (image) { return image.id !== id; });
      if (!m.images.length) delete m.images;
      saveSessions();
      renderMessages();
      discardImages(removed);
      return;
    }
  }
  function clearDraftImages() {
    var removed = state.draftImages;
    state.draftImages = [];
    renderDraftImages();
    discardImages(removed);
  }
  function restoreDraftImages(images, text) {
    var seen = new Set(), merged = [];
    (images || []).concat(state.draftImages).forEach(function (image) {
      var ref = imageRef(image);
      if (ref && !seen.has(ref.id)) { seen.add(ref.id); merged.push(ref); }
    });
    var dropped = merged.slice(MAX_DRAFT_IMAGES);
    state.draftImages = merged.slice(0, MAX_DRAFT_IMAGES);
    if (text && !$('input').value) $('input').value = text;
    autoGrow();
    renderDraftImages();
    discardImages(dropped);
  }
  async function addImageFiles(files) {
    if (state.busy || state.sending || state.imageBusy) return;
    var list = Array.prototype.slice.call(files || []);
    list = list.filter(function (file) {
      return (file.type && file.type.indexOf('image/') === 0) || /\.(png|jpe?g|webp|gif|bmp|svg)$/i.test(file.name || '');
    });
    if (!list.length) { setStatus(t('chat.imageType')); return; }
    var room = MAX_DRAFT_IMAGES - state.draftImages.length;
    if (list.length > room) setStatus(t('chat.imageLimit'));
    list = list.slice(0, Math.max(0, room));
    state.imageBusy = true;
    renderDraftImages();
    setBusy(state.busy);
    var added = 0, pendingId = null;
    try {
      for (var i = 0; i < list.length; i++) {
        var small = await shrinkImageFile(list[i]);
        var ref = imageRef({
          id: uid(), name: list[i].name || 'image', mime: 'image/jpeg',
          width: small.width, height: small.height, size: small.blob.size, createdAt: Date.now()
        });
        pendingId = ref.id;
        await ensureImageCapacity(small.blob.size);
        await imageDbWrite(function (store) {
          store.put({ id: ref.id, name: ref.name, mime: ref.mime, width: ref.width,
                      height: ref.height, size: ref.size, createdAt: ref.createdAt,
                      blob: small.blob });
        });
        await cacheImage({ id: ref.id, name: ref.name, mime: ref.mime,
                           size: ref.size, blob: small.blob });
        state.draftImages.push(ref);
        pendingId = null;
        added++;
        renderDraftImages();
      }
      if (added) setStatus(t('chat.imageAdded') + ' × ' + added);
    } catch (e) {
      if (pendingId) discardImages([{ id: pendingId }]);
      setStatus(t(e && e.code === 'image-full' ? 'chat.imageFull' : 'chat.imageFailed') +
                (e && e.code === 'image-full' ? '' : String(e && e.message ? e.message : e)));
    } finally {
      state.imageBusy = false;
      renderDraftImages();
      setBusy(state.busy);
    }
  }

  /* ------------------------------------------------------------------ */
  /* i18n — instant, no reload.                                          */
  /*                                                                     */
  /* Static text lives in the document as `data-i18n="key"`, and dynamic  */
  /* text produced by this file goes through `t(key)`. Both read the same */
  /* table, so adding a language is one object, not a second DOM pass.    */
  /* The saved choice is merged into prefs by loadPrefs(); nothing here   */
  /* talks to the server — the console chrome is entirely client-side.   */
  /* ------------------------------------------------------------------ */
  var I18N = {
    zh: {
      'doc.title': 'cwick · 本地对话',
      'lang.title': '界面语言 / Interface language',
      'nav.chat': '对话', 'nav.apps': '应用', 'nav.models': '模型',
      'nav.logs': '日志', 'nav.settings': '设置',
      'rail.sessions': '会话', 'rail.newChat': '新建会话 (Ctrl/⌘+K)',
      'rail.memory': '常驻内存',
      'status.connecting': '连接中', 'status.ready': '就绪',
      'status.tokS': '上一次生成的解码速度',
      'view.chat.title': '对话',
      'view.chat.sub': '流式输出 · 思考强度可控 · 同源调用本机 OpenAI 兼容接口',
      'view.apps.title': '应用',
      'view.apps.sub': '内置 H5 小应用 · 直接调用本机同源 OpenAI 兼容接口',
      'view.models.title': '模型',
      'view.models.sub': '本地权重 · 下载进度与断点续传 · 从模型目录检索更多权重',
      'view.logs.title': '日志',
      'view.logs.sub': '进程自身的 stdout / stderr · 和启动它的终端看到同一份输出',
      'view.settings.title': '设置',
      'view.settings.sub': '监控趋势与模型信息 · 自定义生成参数 · 切换本地模型 · 运行配置只读',
      'chat.emptyTitle': '在这台 Mac 上直接对话',
      'chat.emptyBody': '页面由 cwick 自己托管，请求发往同源的',
      'chat.emptyBody2': '、',
      'chat.emptyBody3': ' 或 ',
      'chat.emptyBody4': '（下方"接口"可选）。模型全程本地运行，权重走 SSD 流式，对话不出机器。',
      'chat.ex1': '解释 MoE 与低内存', 'chat.ex2': '写一段 C 代码', 'chat.ex3': '做一份计划',
      'chat.placeholder': '给 cwick 发消息…  Enter 发送 · Shift+Enter 换行',
      'chat.attach': '添加本地图片（可多选，也可粘贴或拖入）',
      'chat.removeImage': '移除图片', 'chat.imageLoading': '正在读取图片…',
      'chat.imageMissing': '图片数据已不存在，请移除后重新添加',
      'chat.imageAdded': '已添加图片', 'chat.imageFailed': '图片处理失败：',
      'chat.imageType': '只支持图片文件', 'chat.imageLimit': '每轮最多添加 8 张图片',
      'chat.imageFull': '本地图片存储已满（上限 64 MiB），请删除旧会话',
      'chat.busyDelete': '当前会话正在生成，暂时不能删除',
      'chat.imageTitle': '图片对话',
      'chat.api': '接口', 'chat.apiTip': '这一轮对话调用的接口：都走同一个引擎，只是请求/响应的信封不同',
      'chat.systemQuestion': '决策问题', 'chat.systemOptions': '选项（逗号分隔）',
      'chat.effort': '思考强度', 'chat.maxOutput': '最大输出', 'chat.stream': '流式',
      'chat.tools': '示例工具', 'chat.toolsTip': '声明示例工具，模型调用时由页面在本地执行',
      'common.refresh': '刷新', 'common.loading': '读取中…',
      'set.models': '本地模型',
      'set.modelsTip': '切换会释放当前常驻权重并清空前缀缓存；生成中会被拒绝（409）',
      'set.image': '图像模型',
      'set.imageTip': '与文本模型相互独立；热切换只加载图像管线，不影响正在常驻的对话模型',
      'set.imageIdle': '未加载', 'set.imageReady': '已加载',
      'set.imageDir': '模型目录', 'set.imageEndpoints': '接口',
      'set.imageLoad': '加载 / 切换到图像模型', 'set.checkModels': '查看模型列表',
      'set.noImage': '未配置图像目录：启动时加 --img DIR，或设置 CWICK_IMAGE_MODEL',
      'set.imageSwitching': '正在热加载图像模型…',
      'set.imageOk': '✓ 图像模型已就绪，两个图像接口均可调用',
      'set.imageFail': '✗ 图像模型加载失败：',
      'hub.ready': '完整', 'hub.partial': '未下载完',
      'hub.unsupported': '不支持', 'hub.archMismatch': '架构未适配，不能切换',
      'hub.switchResident': '设为常驻', 'hub.switchImage': '切换到图像模型',
      'hub.kindImage': '图像模型', 'hub.huge': '超大权重',
      'hub.switchVideo': '切换到视频模型', 'hub.kindVideo': '视频模型',
      'set.videoTip': '视频管线独占常驻槽位：热切换会先释放文本/语音/图像权重，因为采样阶段要映射 62 GiB  transformer'
    },
    en: {
      'doc.title': 'cwick · Local Chat',
      'lang.title': 'Interface language',
      'nav.chat': 'Chat', 'nav.apps': 'Apps', 'nav.models': 'Models',
      'nav.logs': 'Logs', 'nav.settings': 'Settings',
      'rail.sessions': 'Sessions', 'rail.newChat': 'New chat (Ctrl/⌘+K)',
      'rail.memory': 'Resident memory',
      'status.connecting': 'Connecting', 'status.ready': 'Ready',
      'status.tokS': 'Decoding speed of the last generation',
      'view.chat.title': 'Chat',
      'view.chat.sub': 'Streaming · adjustable reasoning effort · same-origin calls to the local OpenAI-compatible API',
      'view.apps.title': 'Apps',
      'view.apps.sub': 'Built-in H5 apps · same-origin calls to the local OpenAI-compatible API',
      'view.models.title': 'Models',
      'view.models.sub': 'Local weights · download progress and resume · browse more weights from the model hub',
      'view.logs.title': 'Logs',
      'view.logs.sub': 'The process stdout / stderr · exactly what the launching terminal sees',
      'view.settings.title': 'Settings',
      'view.settings.sub': 'Metric trends and model info · generation preferences · switch local models · read-only runtime config',
      'chat.emptyTitle': 'Talk to the model on this Mac',
      'chat.emptyBody': 'This page is served by cwick itself. Requests go to the same-origin',
      'chat.emptyBody2': ',',
      'chat.emptyBody3': ' or ',
      'chat.emptyBody4': '(pick the API below). The model runs entirely on this machine, weights stream from SSD, and the conversation never leaves it.',
      'chat.ex1': 'Explain MoE and low memory', 'chat.ex2': 'Write a C function', 'chat.ex3': 'Make a plan',
      'chat.placeholder': 'Message cwick…  Enter to send · Shift+Enter for a new line',
      'chat.attach': 'Add local images (multiple, paste, or drag and drop)',
      'chat.removeImage': 'Remove image', 'chat.imageLoading': 'Loading image…',
      'chat.imageMissing': 'Image data is gone; remove it and add it again',
      'chat.imageAdded': 'Image added', 'chat.imageFailed': 'Image processing failed: ',
      'chat.imageType': 'Only image files are supported', 'chat.imageLimit': 'Up to 8 images per turn',
      'chat.imageFull': 'Local image storage is full (64 MiB limit); delete older chats',
      'chat.busyDelete': 'The current chat is generating and cannot be deleted yet',
      'chat.imageTitle': 'Image chat',
      'chat.api': 'API', 'chat.apiTip': 'The endpoint this turn uses: all reach the same engine, only the request/response envelope differs',
      'chat.systemQuestion': 'Decision question', 'chat.systemOptions': 'Options (comma separated)',
      'chat.effort': 'Reasoning', 'chat.maxOutput': 'Max output', 'chat.stream': 'Stream',
      'chat.tools': 'Sample tools', 'chat.toolsTip': 'Declare sample tools; the page executes them locally when the model calls them',
      'common.refresh': 'Refresh', 'common.loading': 'Loading…',
      'set.models': 'Local models',
      'set.modelsTip': 'Switching releases the resident weights and clears the prefix cache; refused (409) while generating',
      'set.image': 'Image model',
      'set.imageTip': 'Independent of the text model; a hot switch loads only the image pipeline and leaves the resident chat model untouched',
      'set.imageIdle': 'Not loaded', 'set.imageReady': 'Loaded',
      'set.imageDir': 'Model directory', 'set.imageEndpoints': 'Endpoints',
      'set.imageLoad': 'Load / switch to image model', 'set.checkModels': 'View model list',
      'set.noImage': 'No image directory configured: pass --img DIR at startup, or set CWICK_IMAGE_MODEL',
      'set.imageSwitching': 'Hot-loading the image model…',
      'set.imageOk': 'Image model ready; both image endpoints are available',
      'set.imageFail': 'Failed to load the image model: ',
      'hub.ready': 'Complete', 'hub.partial': 'Incomplete',
      'hub.unsupported': 'Unsupported', 'hub.archMismatch': 'No matching kernel',
      'hub.switchResident': 'Make resident', 'hub.switchImage': 'Switch to image model',
      'hub.kindImage': 'Image model', 'hub.huge': 'Huge weights',
      'hub.switchVideo': 'Switch to video model', 'hub.kindVideo': 'Video model',
      'set.videoTip': 'The video pipeline owns the resident slot: a hot switch releases the text/voice/image weights first, because sampling maps the 62 GiB transformer'
    }
  };
  function t(key) {
    var tab = I18N[state.prefs.lang] || I18N.zh;
    return tab[key] != null ? tab[key] : (I18N.zh[key] != null ? I18N.zh[key] : key);
  }
  /* Repaint every tagged node. Called on boot and on language change. */
  function applyI18n() {
    var lang = state.prefs.lang === 'en' ? 'en' : 'zh';
    document.documentElement.lang = lang === 'en' ? 'en' : 'zh-CN';
    var txt = document.querySelectorAll('[data-i18n]');
    for (var i = 0; i < txt.length; i++)
      txt[i].textContent = t(txt[i].getAttribute('data-i18n'));
    var ph = document.querySelectorAll('[data-i18n-ph]');
    for (var j = 0; j < ph.length; j++)
      ph[j].setAttribute('placeholder', t(ph[j].getAttribute('data-i18n-ph')));
    var ti = document.querySelectorAll('[data-i18n-title]');
    for (var k = 0; k < ti.length; k++)
      ti[k].setAttribute('title', t(ti[k].getAttribute('data-i18n-title')));
    var btns = document.querySelectorAll('#langSwitch .lang-btn');
    for (var m = 0; m < btns.length; m++) {
      var on = btns[m].getAttribute('data-lang') === lang;
      btns[m].classList.toggle('is-active', on);
      btns[m].setAttribute('aria-pressed', on ? 'true' : 'false');
    }
  }
  /* Switching language must not blank the user's own session labels. Those
   * are re-rendered from data, so only the static chrome needs repainting. */
  function setLang(lang) {
    if (lang !== 'en' && lang !== 'zh') return;
    state.prefs.lang = lang;
    savePrefs();
    applyI18n();
    /* Re-capture the translated empty state and repaint it if visible, so
     * switching language cannot leave stale text in an active conversation. */
    if ($('emptyState')) EMPTY_HTML = $('emptyState').outerHTML;
    renderMessages();
    renderDraftImages();
    renderModelInfoStrings();
    /* Content fade-in: re-trigger the animation on the main panel so the
     * change reads as a transition, not a jump. */
    var main = document.querySelector('.main');
    if (main) {
      main.classList.remove('lang-enter');
      void main.offsetWidth;              /* force reflow to restart CSS anim */
      main.classList.add('lang-enter');
    }
    /* Focus management: move keyboard focus to the content area so a
     * screen-reader or keyboard user lands where the text just changed. */
    var title = $('viewTitle');
    if (title) { title.setAttribute('tabindex', '-1'); try { title.focus({ preventScroll: true }); } catch (e) { title.focus(); } }
  }
  /* Dynamic strings that the i18n pass cannot see because they are built in
   * JS. Kept in one place so setLang() can refresh them in one call. */
  function renderModelInfoStrings() {
    if (state.view === 'settings') {
      loadImageModelStatus();
    }
  }

  function loadPrefs() {
    try {
      var raw = localStorage.getItem(LS_PREFS);
      if (raw) {
        var p = JSON.parse(raw);
        /* Older builds kept an external endpoint next to these preferences.
           Remove it so no page on this origin can pick the stale address up. */
        var stale = ['apiBase', 'apiKey', 'apiModel'], dirty = false;
        for (var i = 0; i < stale.length; i++) {
          if (stale[i] in p) { delete p[stale[i]]; dirty = true; }
        }
        if (dirty) { try { localStorage.setItem(LS_PREFS, JSON.stringify(p)); } catch (e2) {} }
        for (var k in state.prefs) if (k in p) state.prefs[k] = p[k];
      }
    } catch (e) { /* private mode: defaults are fine */ }
  }
  function savePrefs() {
    try { localStorage.setItem(LS_PREFS, JSON.stringify(state.prefs)); } catch (e) {}
  }

  /* Every request goes to the server that served this page. The session
     cookie is the only credential the browser needs; the named API keys in
     the settings view are for other clients calling *this* server. */
  function apiBaseUrl() { return location.origin + '/v1'; }
  function apiUrl(path) { return apiBaseUrl() + path; }
  function apiHeaders(json) {
    var h = {};
    if (json) h['Content-Type'] = 'application/json';
    return h;
  }
  function apiModel() { return state.modelId || 'glm-5.3-flash'; }
  /* A protected endpoint answered 401: send the browser to the login page and
     come back to exactly this view (hash included). API clients never see this
     path because they can read the 401 and mint a Bearer token instead. */
  function authRedirect(res) {
    if (!res || res.status !== 401) return false;
    var next = encodeURIComponent(location.pathname + location.search + location.hash);
    location.href = '/login?next=' + next;
    return true;
  }

  async function loadAuthState() {
    var el = $('authState');
    if (!el) return;
    try {
      var r = await fetch('/v1/auth/status', { cache: 'no-store' });
      if (authRedirect(r)) return;
      var j = await r.json();
      if (j.enabled) {
        el.className = 'note ok';
        el.textContent = '单用户登录已启用。当前会话已认证；退出后需要重新输入密码。';
        if ($('logoutBtn')) $('logoutBtn').disabled = false;
      } else {
        el.className = 'note';
        el.textContent = '登录未启用（默认仅监听 127.0.0.1）。使用 --password 或 --lan 可要求单用户密码。';
        if ($('logoutBtn')) $('logoutBtn').disabled = true;
      }
    } catch (e) {
      el.className = 'note bad';
      el.textContent = '无法读取 /v1/auth/status';
    }
  }

  /* ------------------------------------------------------------------ */
  /* API keys                                                            */
  /* ------------------------------------------------------------------ */

  function keyTime(ts) {
    if (!ts) return '从未使用';
    try {
      return new Date(ts * 1000).toLocaleString('zh-CN', { hour12: false });
    } catch (e) { return String(ts); }
  }

  async function loadKeys() {
    var box = $('keyList');
    if (!box) return;
    try {
      var r = await fetch('/v1/auth/keys', { cache: 'no-store' });
      if (authRedirect(r)) return;
      if (r.status === 404) {
        box.innerHTML = '<div class="muted">登录未启用，因此没有密钥。用 <code>--api-key NAME</code>' +
          ' 或 <code>--password</code> 启动后可以创建。</div>';
        return;
      }
      if (r.status === 401) {
        box.innerHTML = '<div class="muted bad">需要登录会话才能查看密钥（密钥本身不能管理密钥）。</div>';
        return;
      }
      if (!r.ok) throw new Error('HTTP ' + r.status);
      var j = await r.json();
      var keys = (j && j.keys) || [];
      if (!keys.length) {
        box.innerHTML = '<div class="muted">还没有密钥。新建一个，把它粘贴到你的 Agent 平台里。</div>';
        return;
      }
      var html = '';
      for (var i = 0; i < keys.length; i++) {
        var k = keys[i];
        html += '<div class="key-row">' +
          '<div class="key-main"><b>' + esc(k.name || 'key') + '</b>' +
          '<code class="mono">' + esc(k.prefix || '') + '…</code></div>' +
          '<div class="key-meta muted mono">创建 ' + keyTime(k.created_at) +
          ' · ' + (k.last_used_at ? '最近使用 ' + keyTime(k.last_used_at) : '未使用') +
          '</div>' +
          '<button class="min-btn danger" type="button" data-revoke="' +
          esc(k.id || '') + '">撤销</button></div>';
      }
      box.innerHTML = html;
    } catch (e) {
      box.innerHTML = '<div class="muted bad">无法读取 /v1/auth/keys：' + esc(String(e.message || e)) + '</div>';
    }
  }

  async function createKey() {
    var nameEl = $('keyName');
    var name = nameEl ? nameEl.value.trim() : '';
    var box = $('keyList');
    try {
      var r = await fetch('/v1/auth/keys', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: name })
      });
      if (authRedirect(r)) return;
      var j = await r.json().catch(function () { return {}; });
      if (!r.ok) {
        if (box) box.innerHTML = '<div class="muted bad">' +
          esc((j.error && j.error.message) || ('HTTP ' + r.status)) + '</div>';
        return;
      }
      var sec = $('keySecret'), txt = $('keySecretText');
      if (sec && txt) {
        txt.textContent = j.secret || '';
        sec.hidden = false;
      }
      if (nameEl) nameEl.value = '';
      loadKeys();
    } catch (e) {
      if (box) box.innerHTML = '<div class="muted bad">创建失败：' + esc(String(e.message || e)) + '</div>';
    }
  }

  async function revokeKey(id) {
    if (!id) return;
    if (!window.confirm('撤销这个密钥？正在使用它的 Agent 会立即收到 401。')) return;
    try {
      var r = await fetch('/v1/auth/keys?id=' + encodeURIComponent(id), { method: 'DELETE' });
      if (authRedirect(r)) return;
      /* A revoked key must stop looking usable: leaving its plaintext on screen
       * next to "还没有密钥" invites copying a secret that now returns 401. */
      var sec = $('keySecret');
      if (sec) sec.hidden = true;
    } catch (e) { /* the list refresh below shows whatever happened */ }
    loadKeys();
  }

  async function copyKeySecret() {
    var txt = $('keySecretText');
    if (!txt) return;
    var text = txt.textContent || '';
    try {
      if (navigator.clipboard) await navigator.clipboard.writeText(text);
      else {
        var t = document.createElement('textarea');
        t.value = text; document.body.appendChild(t); t.select();
        document.execCommand('copy'); t.remove();
      }
      var b = $('keyCopy');
      if (b) { b.textContent = '已复制'; setTimeout(function () { b.textContent = '复制'; }, 1200); }
    } catch (e) {}
  }

  function loadSessions() {
    try {
      var raw = localStorage.getItem(LS_SESSIONS);
      if (raw) state.sessions = JSON.parse(raw) || [];
    } catch (e) { state.sessions = []; }
    if (!Array.isArray(state.sessions)) state.sessions = [];
    state.sessions = state.sessions.filter(function (s) {
      return s && typeof s === 'object' && Array.isArray(s.messages);
    }).map(storedSession);
  }
  function saveSessions() {
    /* Only image references reach localStorage; the blobs live in IndexedDB.
       30 sessions plus a 64 MiB image-store ceiling keeps both stores bounded. */
    var dropped = [];
    while (state.sessions.length > 30) dropped.push(state.sessions.pop());
    if (dropped.length) {
      if (state.currentId && !state.sessions.some(function (s) { return s.id === state.currentId; }))
        state.currentId = state.sessions[0] ? state.sessions[0].id : null;
      discardImages(dropped.reduce(function (all, s) { return all.concat(sessionImages(s)); }, []));
    }
    try {
      localStorage.setItem(LS_SESSIONS, JSON.stringify(state.sessions.map(storedSession)));
    } catch (e) {
      /* Quota exceeded: drop the tail and retry once, then give up quietly. */
      var keep = Math.max(1, Math.floor(state.sessions.length / 2));
      dropped = state.sessions.slice(keep);
      state.sessions = state.sessions.slice(0, keep);
      if (state.currentId && !state.sessions.some(function (s) { return s.id === state.currentId; }))
        state.currentId = state.sessions[0] ? state.sessions[0].id : null;
      discardImages(dropped.reduce(function (all, s) { return all.concat(sessionImages(s)); }, []));
      try { localStorage.setItem(LS_SESSIONS, JSON.stringify(state.sessions.map(storedSession))); } catch (e2) {}
    }
  }

  function current() {
    for (var i = 0; i < state.sessions.length; i++) {
      if (state.sessions[i].id === state.currentId) return state.sessions[i];
    }
    return null;
  }
  function newSession() {
    clearDraftImages();
    var s = { id: uid(), title: '新会话', messages: [], createdAt: Date.now() };
    state.sessions.unshift(s);
    state.currentId = s.id;
    saveSessions();
    renderSessions();
    renderMessages();
    return s;
  }

  /* ------------------------------------------------------------------ */
  /* markdown (escaped first: model output is untrusted text)            */
  /* ------------------------------------------------------------------ */
  function inlineMd(s) {
    s = s.replace(/`([^`]+)`/g, function (m, code) {
      return '<code class="inline">' + code + '</code>';
    });
    s = s.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    s = s.replace(/(^|[^*])\*([^*\n]+)\*(?!\*)/g, '$1<em>$2</em>');
    s = s.replace(/(https?:\/\/[^\s<)"]+)/g, function (m, url) {
      return '<a href="' + url + '" target="_blank" rel="noreferrer">' + url + '</a>';
    });
    return s;
  }

  /* A fenced block is worth more than reading: the same bytes can be copied
     out, saved to disk, and — when they are a page — actually run. `code` is
     already escaped (renderMd escapes everything first), so the buttons read
     the source back out of `code.textContent` rather than keeping a copy. */
  var LANG_EXT = {
    js: 'js', javascript: 'js', mjs: 'js', ts: 'ts', typescript: 'ts', json: 'json',
    py: 'py', python: 'py', sh: 'sh', bash: 'sh', shell: 'sh', zsh: 'sh',
    c: 'c', h: 'h', cpp: 'cpp', cc: 'cpp', go: 'go', rs: 'rs', rust: 'rs',
    java: 'java', sql: 'sql', css: 'css', scss: 'css', yml: 'yaml', yaml: 'yaml',
    md: 'md', markdown: 'md', txt: 'txt', html: 'html', htm: 'html',
    xhtml: 'html', svg: 'svg', xml: 'xml'
  };

  function isHtmlBlock(lang, code) {
    if (/^(html?|xhtml|svg|xml)$/i.test(lang || '')) return true;
    /* no language given: recognise a page by its first tag (escaped form) */
    return /^\s*&lt;(!doctype\s+html|html[\s>])/i.test(code);
  }

  function codeBlock(lang, code) {
    var name = esc(lang || 'text');
    var html = isHtmlBlock(lang, code);
    var acts = '';
    if (html) acts += '<button type="button" data-act="run">运行</button>';
    acts += '<button type="button" data-act="dl">下载</button>' +
            '<button type="button" data-act="copy">复制</button>';
    return '<div class="code-block"' + (html ? ' data-html="1"' : '') +
      ' data-lang="' + name + '"><div class="code-head"><span>' + name +
      '</span><span class="code-acts">' + acts + '</span>' +
      '</div><pre><code>' + code + '</code></pre></div>';
  }

  function blockText(block) {
    var c = block && block.querySelector('code');
    return c ? c.textContent : '';
  }
  function extFor(block) {
    var lang = String((block && block.dataset && block.dataset.lang) || 'text').toLowerCase();
    if (LANG_EXT[lang]) return LANG_EXT[lang];
    return /^[a-z0-9]{1,6}$/.test(lang) ? lang : 'txt';
  }
  function flash(btn, msg) {
    if (!btn) return;
    var old = btn.dataset.label || btn.textContent;
    btn.dataset.label = old;
    btn.textContent = msg;
    setTimeout(function () { btn.textContent = old; }, 1200);
  }
  function copyText(text, btn) {
    if (!navigator.clipboard) { flash(btn, '不可用'); return; }
    navigator.clipboard.writeText(text).then(
      function () { flash(btn, '已复制'); },
      function () { flash(btn, '失败'); });
  }
  function downloadText(text, filename) {
    var url = URL.createObjectURL(new Blob([text], { type: 'text/plain;charset=utf-8' }));
    var a = document.createElement('a');
    a.href = url;
    a.download = filename || 'snippet.txt';
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(function () { URL.revokeObjectURL(url); }, 4000);
  }

  /* Running a page the model wrote. The iframe is sandboxed without
     allow-same-origin, so it cannot read this origin's storage, cookies or the
     session token — it can still run scripts and show itself. The preview
     lives in a modal rather than inside the bubble: the bubble is re-rendered
     on every streamed token, and an inline iframe would reload on each one. */
  var runState = { el: null, src: '' };

  function runModal() {
    if (runState.el) return runState.el;
    var wrap = document.createElement('div');
    wrap.className = 'runmodal';
    wrap.hidden = true;
    wrap.innerHTML =
      '<div class="rm-card">' +
        '<div class="rm-head"><span class="rm-title">HTML 预览 · 沙箱</span>' +
          '<span class="rm-acts">' +
            '<button type="button" data-act="rm-copy">复制</button>' +
            '<button type="button" data-act="rm-dl">下载</button>' +
            '<button type="button" data-act="rm-reload">重新运行</button>' +
            '<button type="button" data-act="rm-close">✕</button>' +
          '</span></div>' +
        '<div class="rm-body"><iframe title="HTML 预览"></iframe></div>' +
      '</div>';
    wrap.addEventListener('click', function (e) {
      if (e.target === wrap) { closeRun(); return; }
      var b = e.target.closest('button[data-act]');
      if (!b) return;
      var act = b.dataset.act;
      if (act === 'rm-close') closeRun();
      else if (act === 'rm-reload') loadRun(runState.src);
      else if (act === 'rm-copy') copyText(runState.src, b);
      else if (act === 'rm-dl') downloadText(runState.src, 'cwick-preview.html');
    });
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape' && runState.el && !runState.el.hidden) closeRun();
    });
    document.body.appendChild(wrap);
    runState.el = wrap;
    return wrap;
  }
  /* A used iframe keeps its document alive; swapping the node is the only
     reliable way to stop whatever timers the snippet left running. */
  function freshFrame() {
    var m = runModal();
    var f = m.querySelector('iframe');
    var nf = document.createElement('iframe');
    nf.title = 'HTML 预览';
    nf.setAttribute('sandbox', 'allow-scripts allow-modals allow-forms allow-popups');
    nf.setAttribute('referrerpolicy', 'no-referrer');
    f.parentNode.replaceChild(nf, f);
    return nf;
  }
  function loadRun(src) {
    runState.src = src;
    var m = runModal();
    m.hidden = false;
    freshFrame().srcdoc = src;
  }
  function closeRun() {
    if (!runState.el) return;
    runState.el.hidden = true;
    freshFrame();
  }

  function renderMd(src) {
    var lines = esc(src).split('\n');
    var out = '', para = [], list = null, i = 0;

    function flushPara() {
      if (para.length) { out += '<p>' + inlineMd(para.join(' ')) + '</p>'; para = []; }
    }
    function closeList() { if (list) { out += '</' + list + '>'; list = null; } }

    while (i < lines.length) {
      var line = lines[i];

      if (/^\s*```/.test(line)) {
        flushPara(); closeList();
        var lang = line.replace(/^\s*```/, '').trim();
        var buf = [];
        i++;
        while (i < lines.length && !/^\s*```/.test(lines[i])) { buf.push(lines[i]); i++; }
        i++;
        out += codeBlock(lang, buf.join('\n'));
        continue;
      }
      if (!line.trim()) { flushPara(); closeList(); i++; continue; }

      var h = line.match(/^(#{1,4})\s+(.*)$/);
      if (h) { flushPara(); closeList(); out += '<h3>' + inlineMd(h[2]) + '</h3>'; i++; continue; }

      var q = line.match(/^&gt;\s?(.*)$/);
      if (q) { flushPara(); closeList(); out += '<blockquote>' + inlineMd(q[1]) + '</blockquote>'; i++; continue; }

      var ul = line.match(/^\s*[-*+]\s+(.*)$/);
      var ol = line.match(/^\s*\d+[.)]\s+(.*)$/);
      if (ul || ol) {
        flushPara();
        var want = ul ? 'ul' : 'ol';
        if (list !== want) { closeList(); out += '<' + want + '>'; list = want; }
        out += '<li>' + inlineMd((ul || ol)[1]) + '</li>';
        i++;
        continue;
      }
      if (/^\s*(---+|___+|\*\*\*+)\s*$/.test(line)) {
        flushPara(); closeList(); out += '<hr>'; i++; continue;
      }
      para.push(line);
      i++;
    }
    flushPara(); closeList();
    return out || '<p></p>';
  }

  /* Parsed calls, one card each. Split from the wrapper so the streaming
     painter can drop the same cards into a container it owns. */
  function toolCallItems(calls) {
    if (!calls || !calls.length) return '';
    var h = '';
    for (var i = 0; i < calls.length; i++) {
      var tc = calls[i];
      h += '<div class="toolcall"><div class="tc-head"><span class="tc-name">' +
        esc(tc.name) + '</span><span class="tc-kind">工具调用' +
        (tc.result != null ? ' · 已执行' : '') + '</span></div>' +
        '<div class="tc-args">' + esc(tc.arguments || '{}') + '</div>' +
        (tc.result != null ? '<div class="tc-result">↳ ' + esc(tc.result) + '</div>' : '') +
        '</div>';
    }
    return h;
  }
  function toolCallsHtml(calls) {
    return calls && calls.length ? '<div class="tools">' + toolCallItems(calls) + '</div>' : '';
  }
  /* Fold OpenAI streaming `tool_calls` chunks into a message. One chunk
     opens a card (id/name/arguments:""), later chunks sharing its index
     append to the arguments, and the closing chunk repeats the id without
     discarding what was streamed. Pure so it can be checked without a DOM. */
  function applyToolCallDeltas(msg, chunks) {
    var list = msg.toolCalls || (msg.toolCalls = []);
    chunks.forEach(function (tc, k) {
      var fn = tc.function || {};
      /* A streaming shard names its index; a whole array at the finish has
         no index, so its position in that array is the call number. */
      var idx = typeof tc.index === 'number' ? tc.index : k;
      var prev = list[idx];
      list[idx] = {
        id: tc.id || (prev && prev.id) || '',
        name: fn.name || (prev && prev.name) || '',
        arguments: ((prev && prev.arguments) || '') + (fn.arguments || '')
      };
      if (!prev) {
        msg.toolStream = '';
        msg.toolStreamMuted = true;
      }
    });
    return list;
  }

  /* The block for a call still being written. */
  function toolStreamHtml() {
    return '<div class="toolstream"><div class="ts-head"><span class="ts-dot"></span>' +
      '正在生成工具调用…<span class="ts-hint">arguments</span></div>' +
      '<pre class="ts-body"></pre></div>';
  }

  /* ------------------------------------------------------------------ */
  /* chat rendering                                                      */
  /* ------------------------------------------------------------------ */
  function msgNode(m, streaming) {
    var wrap = document.createElement('div');
    wrap.className = 'msg ' + (m.role === 'user' ? 'user' : 'assistant');

    if (m.role === 'user') {
      var images = Array.isArray(m.images) ? m.images : [];
      wrap.innerHTML = '<div class="body"><div class="bubble-user' + (images.length ? ' has-images' : '') + '">' +
        (images.length ? '<div class="msg-images">' +
          images.map(function (image) { return imageFigureHtml(image, false); }).join('') + '</div>' : '') +
        '<div class="msg-text">' + esc(m.content) + '</div></div></div>';
      wrap.querySelectorAll('[data-remove-history-image]').forEach(function (button) {
        button.disabled = state.busy || state.sending;
      });
      paintImageFigures(wrap, images);
      return wrap;
    }

    var html = '<div class="avatar">c</div><div class="body">';
    if (m.reasoning) {
      var chars = m.reasoning.length;
      html += '<details class="think"' + (streaming && !m.content ? ' open' : '') + '>' +
        '<summary><span class="chev">▶</span><span>' +
        (streaming && !m.content ? '正在思考…' : '思考过程') +
        '</span><span class="tag">' + fmtNum(chars) + ' 字</span></summary>' +
        '<div class="think-body" data-role="think">' + esc(m.reasoning) + '</div></details>';
    }
    html += toolCallsHtml(m.toolCalls);
    html += '<div class="content" data-role="content">' +
      (m.content ? renderMd(m.content)
                 : (streaming ? ''
                    : '<p class="muted">' + (m.reasoning
                        ? '（本轮只产生了思考内容，没有正文：多半是在思考中到达了输出上限，调大“最大输出”再试）'
                        : '（空响应）') + '</p>')) +
      (streaming ? '<span class="caret"></span>' : '') + '</div>';
    if (m.stats || streaming) {
      html += '<div class="msg-stats" data-role="stats">' + (m.stats || '') + '</div>';
    }
    if (m.error) html += '<div class="err">' + esc(m.error) + '</div>';
    html += '</div>';
    wrap.innerHTML = html;
    return wrap;
  }

  function nearBottom() {
    var el = $('messages');
    return el.scrollHeight - el.scrollTop - el.clientHeight < 120;
  }
  function scrollDown(force) {
    var el = $('messages');
    if (force || nearBottom()) el.scrollTop = el.scrollHeight;
  }

  function renderMessages() {
    var box = $('messages');
    var s = current();
    if (!s || !s.messages.length) {
      box.innerHTML = EMPTY_HTML;
      return;
    }
    box.innerHTML = '';
    for (var i = 0; i < s.messages.length; i++) {
      /* `role:"tool"` entries exist for the API (the model has to see its own
         call and the result), but they are already shown inside the
         assistant bubble as the "↳ result" line. Rendering them again would
         leave empty-looking bubbles after every tool round. */
      if (s.messages[i].role === 'tool') continue;
      box.appendChild(msgNode(s.messages[i], false));
    }
    scrollDown(true);
  }

  function renderSessions() {
    var box = $('sessionList');
    box.innerHTML = '';
    if (!state.sessions.length) {
      box.innerHTML = '<div class="session" style="cursor:default;color:var(--faint)">还没有会话</div>';
      return;
    }
    state.sessions.forEach(function (s) {
      var row = document.createElement('div');
      row.className = 'session' + (s.id === state.currentId ? ' is-active' : '');
      row.dataset.id = s.id;
      row.innerHTML = '<span class="name">' + esc(s.title || '新会话') + '</span>' +
        '<button class="del" title="删除会话">×</button>';
      box.appendChild(row);
    });
  }

  /* ------------------------------------------------------------------ */
  /* API                                                                 */
  /* ------------------------------------------------------------------ */
  function ApiError(status, body) {
    this.status = status;
    this.body = body;
    this.message = body || ('HTTP ' + status);
  }
  ApiError.prototype = Object.create(Error.prototype);

  function apiErrorText(err) {
    var b = err && err.body ? err.body : '';
    try {
      var j = JSON.parse(b);
      if (j && j.error && j.error.message) b = j.error.message;
    } catch (e) {}
    if (err && err.status === 429) {
      return '生成槽正被占用（429）：' + b + '\n这个服务端按 FIFO 单槽排队，稍等片刻再发一次。';
    }
    if (err && err.status) return 'HTTP ' + err.status + '：' + b;
    return b || String(err);
  }

  var TOOL_DEFS = [
    {
      type: 'function',
      function: {
        name: 'get_current_time',
        description: '返回本机当前时间（ISO 8601，含时区）。',
        parameters: { type: 'object', properties: {}, required: [] }
      }
    },
    {
      type: 'function',
      function: {
        name: 'add_numbers',
        description: '把两个数字相加并返回结果。',
        parameters: {
          type: 'object',
          properties: { a: { type: 'number' }, b: { type: 'number' } },
          required: ['a', 'b']
        }
      }
    },
    {
      type: 'function',
      function: {
        name: 'system_info',
        description: '返回浏览器的语言、时区、逻辑核数与设备内存（GiB）。',
        parameters: { type: 'object', properties: {}, required: [] }
      }
    }
  ];

  function runTool(name, argsJson) {
    var args = {};
    try { args = JSON.parse(argsJson || '{}') || {}; } catch (e) {}
    if (name === 'get_current_time') {
      var d = new Date();
      var off = -d.getTimezoneOffset();
      var sign = off >= 0 ? '+' : '-';
      var pad = function (n) { return String(Math.floor(Math.abs(n))).padStart(2, '0'); };
      return d.getFullYear() + '-' + String(d.getMonth() + 1).padStart(2, '0') + '-' +
        String(d.getDate()).padStart(2, '0') + 'T' +
        String(d.getHours()).padStart(2, '0') + ':' +
        String(d.getMinutes()).padStart(2, '0') + ':' +
        String(d.getSeconds()).padStart(2, '0') + sign + pad(off / 60) + ':' + pad(off % 60);
    }
    if (name === 'add_numbers') {
      var a = Number(args.a || 0), b = Number(args.b || 0);
      return String(a + b);
    }
    if (name === 'system_info') {
      return JSON.stringify({
        language: navigator.language,
        timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
        cores: navigator.hardwareConcurrency || null,
        memoryGiB: navigator.deviceMemory || null,
        platform: navigator.platform
      });
    }
    return 'unknown tool: ' + name;
  }

  /* ---- dialects -----------------------------------------------------
     The console can talk to either OpenAI endpoint. Chat answers in
     `choices[].delta`; Responses answers in typed output items and named
     events. The translation below is deliberately one-way and total: after
     it, a Responses turn is a chat message, so the bubbles, the tool loop
     and the stats line have a single input shape to worry about.

     Pure functions, no `state` and no DOM, so the mapping can be exercised
     on its own against a recorded event stream. */
  function respText(parts) {
    var out = '';
    for (var i = 0; parts && i < parts.length; i++) {
      var p = parts[i] || {};
      if (typeof p.text === 'string') out += p.text;
    }
    return out;
  }
  /* The `usage` object, spelled the chat way: the console's token counters
     and tok/s line were written against prompt/completion. */
  function respUsage(u) {
    if (!u) return null;
    var inTok = u.input_tokens || 0, outTok = u.output_tokens || 0;
    return {
      prompt_tokens: inTok,
      completion_tokens: outTok,
      total_tokens: u.total_tokens || (inTok + outTok),
      completion_tokens_details: {
        reasoning_tokens: (u.output_tokens_details || {}).reasoning_tokens || 0
      }
    };
  }
  function chatChunk(delta, finish, usage) {
    var obj = { choices: [{ index: 0, delta: delta || {}, finish_reason: finish || null }] };
    if (usage) obj.usage = usage;
    return obj;
  }
  function respItemToCall(it) {
    return {
      id: it.call_id,
      type: 'function',
      function: { name: it.name, arguments: it.arguments || '{}' }
    };
  }
  /* A whole buffered `response` -> one chat completion. */
  function respFullToChat(obj) {
    var content = '', reasoning = '', calls = [];
    var items = (obj && obj.output) || [];
    for (var i = 0; i < items.length; i++) {
      var it = items[i] || {};
      if (it.type === 'reasoning') {
        reasoning += respText(it.summary && it.summary.length ? it.summary : it.content);
      } else if (it.type === 'message') {
        content += respText(it.content);
      } else if (it.type === 'function_call') {
        calls.push(respItemToCall(it));
      }
    }
    return {
      choices: [{
        index: 0,
        message: {
          role: 'assistant', content: content,
          reasoning_content: reasoning, tool_calls: calls
        },
        finish_reason: calls.length ? 'tool_calls'
          : (obj && obj.status === 'incomplete' ? 'length' : 'stop')
      }],
      usage: respUsage(obj && obj.usage)
    };
  }
  /* One named event -> one chat chunk, or null when the event carries
     nothing the page draws (created/in_progress/content_part.*). */
  function respEventToChat(ev) {
    var t = (ev && ev.type) || '';
    if (t === 'response.output_text.delta') return chatChunk({ content: ev.delta || '' });
    if (t === 'response.reasoning_summary_text.delta' ||
        t === 'response.reasoning_text.delta') {
      return chatChunk({ reasoning_content: ev.delta || '' });
    }
    /* The item is announced when the model starts the call and its
       arguments arrive as deltas while it is still writing; show the name
       plus each shard, then the parsed card from the completed item. */
    if (t === 'response.output_item.added' && ev.item && ev.item.type === 'function_call') {
      return chatChunk({ cwick_tool: (ev.item.name || '') + '(' });
    }
    if (t === 'response.function_call_arguments.delta') {
      return chatChunk({ cwick_tool: ev.delta || '' });
    }
    if (t === 'response.output_item.done' && ev.item && ev.item.type === 'function_call') {
      return chatChunk({ tool_calls: [respItemToCall(ev.item)] });
    }
    /* `incomplete` is the same envelope with a different verdict (the budget
       ran out); the page draws it like a finished answer and the stats line
       still gets its usage. */
    if (t === 'response.completed' || t === 'response.incomplete') {
      return chatChunk({}, null, respUsage(ev.response && ev.response.usage));
    }
    return null;
  }

  /* ---- the Anthropic dialect ----------------------------------------
     Same idea as the Responses mapping above, with different nouns:
     `content` is an array of typed blocks, a call is `tool_use` and its
     result is a `tool_result` block in the next user turn. Pure functions,
     so the mapping can be exercised against a recorded event stream. */
  function msgFullToChat(obj) {
    var content = '', reasoning = '', calls = [];
    var blocks = (obj && obj.content) || [];
    for (var i = 0; i < blocks.length; i++) {
      var b = blocks[i] || {};
      if (b.type === 'text') content += b.text || '';
      else if (b.type === 'thinking') reasoning += b.thinking || '';
      else if (b.type === 'tool_use') {
        calls.push({
          id: b.id,
          type: 'function',
          function: { name: b.name, arguments: JSON.stringify(b.input || {}) }
        });
      }
    }
    var usage = null;
    if (obj && obj.usage) {
      var inTok = obj.usage.input_tokens || 0, outTok = obj.usage.output_tokens || 0;
      usage = {
        prompt_tokens: inTok, completion_tokens: outTok,
        total_tokens: inTok + outTok
      };
    }
    return {
      choices: [{
        index: 0,
        message: {
          role: 'assistant', content: content,
          reasoning_content: reasoning, tool_calls: calls
        },
        finish_reason: calls.length ? 'tool_calls'
          : (obj && obj.stop_reason === 'max_tokens' ? 'length' : 'stop')
      }],
      usage: usage
    };
  }
  function msgStreamState() { return { in_tokens: 0, tool: null }; }
  /* One named event -> one chat chunk. `st` carries the two things a block
     stream has to remember: the prompt size announced in message_start (the
     terminal message_delta only repeats the output count) and the tool_use
     block whose arguments arrive as `input_json_delta` fragments. */
  function msgEventToChat(ev, st) {
    var t = (ev && ev.type) || '';
    if (t === 'message_start') {
      var u = ev.message && ev.message.usage;
      if (u) st.in_tokens = u.input_tokens || 0;
      return null;
    }
    if (t === 'content_block_start') {
      var b = ev.content_block || {};
      if (b.type === 'tool_use') st.tool = { id: b.id, name: b.name, args: '' };
      return null;
    }
    if (t === 'content_block_delta') {
      var d = ev.delta || {};
      if (d.type === 'text_delta') return chatChunk({ content: d.text || '' });
      if (d.type === 'thinking_delta') return chatChunk({ reasoning_content: d.thinking || '' });
      if (d.type === 'input_json_delta') {
        if (st.tool) st.tool.args += d.partial_json || '';
        /* This dialect *does* stream the call's arguments, so the tool card
           can fill in while the model writes them. */
        return chatChunk({ cwick_tool: d.partial_json || '' });
      }
      return null;
    }
    if (t === 'content_block_stop') {
      if (!st.tool) return null;
      var call = {
        id: st.tool.id, type: 'function',
        function: { name: st.tool.name, arguments: st.tool.args || '{}' }
      };
      st.tool = null;
      return chatChunk({ tool_calls: [call] });
    }
    if (t === 'message_delta') {
      var stop = (ev.delta && ev.delta.stop_reason) || 'end_turn';
      var out = (ev.usage && ev.usage.output_tokens) || 0;
      return chatChunk({},
        stop === 'tool_use' ? 'tool_calls' : stop === 'max_tokens' ? 'length' : 'stop',
        {
          prompt_tokens: st.in_tokens, completion_tokens: out,
          total_tokens: st.in_tokens + out
        });
    }
    return null;   /* message_stop, ping */
  }

  function systemOneFull(data) {
    var answer = data && data.answers && data.answers.decision;
    var text = 'No decision returned';
    if (answer) {
      if (answer.type === 'choice') {
        text = String(answer.choice || '');
        if (answer.probabilities && answer.probabilities[answer.choice] != null)
          text += ' (p=' + Number(answer.probabilities[answer.choice]).toFixed(4) + ')';
      } else if (answer.type === 'score') text = 'score=' + Number(answer.score).toFixed(4);
      else if (answer.type === 'noul') text = 'P(true)=' + Number(answer.noul).toFixed(4);
    }
    var usage = data && data.usage || {};
    return {
      id: data && data.model || 'systemone',
      object: 'chat.completion', created: Math.floor(Date.now() / 1000),
      model: data && data.model || apiModel(),
      choices: [{ index: 0, message: { role: 'assistant', content: text }, finish_reason: 'stop' }],
      usage: {
        prompt_tokens: usage.input_tokens || 0,
        completion_tokens: usage.output_tokens || 0,
        total_tokens: usage.input_tokens || 0
      }
    };
  }

  function normalizeApi(v) {
    return v === 'responses' || v === 'messages' || v === 'systemone' ? v : 'chat';
  }
  function apiKind() { return normalizeApi(state.prefs.api); }
  function updateApiMode() {
    var system = apiKind() === 'systemone';
    $('systemOneFields').hidden = !system;
    $('streamToggle').disabled = system;
    $('toolToggle').disabled = system;
    $('effort').disabled = system;
    $('maxTokens').disabled = system;
  }

  /* One POST. Streaming requests are parsed as SSE; the caller only sees
     callbacks, so the tool loop and the single-shot path share this code. */
  async function postChat(payload, onEvent) {
    var kind = apiKind();
    var path = kind === 'responses' ? '/responses'
             : kind === 'messages' ? '/messages'
             : kind === 'systemone' ? '/systemone'
             : '/chat/completions';
    var mstate = msgStreamState();
    var ctrl = new AbortController();
    state.abort = ctrl;
    var res;
    try {
      res = await fetch(apiUrl(path), {
        method: 'POST',
        headers: apiHeaders(true),
        body: JSON.stringify(payload),
        signal: ctrl.signal
      });
    } catch (e) {
      if (e && e.name === 'AbortError') throw e;
      throw new ApiError(0, '无法连接 cwick：' + (e && e.message ? e.message : e));
    }
    if (authRedirect(res)) throw new ApiError(401, 'authentication required');
    if (!res.ok) {
      var text = '';
      try { text = await res.text(); } catch (e2) {}
      throw new ApiError(res.status, text);
    }
    if (!payload.stream) {
      var data = await res.json();
      if (kind === 'responses') {
        if (data && data.type === 'error') throw new ApiError(0, data.message || '生成失败');
        onEvent({ kind: 'full', data: respFullToChat(data) });
      } else if (kind === 'messages') {
        if (data && data.type === 'error')
          throw new ApiError(0, (data.error && data.error.message) || '生成失败');
        onEvent({ kind: 'full', data: msgFullToChat(data) });
      } else if (kind === 'systemone') {
        if (data && data.type === 'error')
          throw new ApiError(0, (data.error && data.error.message) || '生成失败');
        onEvent({ kind: 'full', data: systemOneFull(data) });
      } else {
        onEvent({ kind: 'full', data: data });
      }
      return;
    }
    var reader = res.body.getReader();
    var dec = new TextDecoder('utf-8');
    var buf = '';
    for (;;) {
      var chunk;
      try { chunk = await reader.read(); } catch (e3) {
        if (e3 && e3.name === 'AbortError') throw e3;
        throw new ApiError(0, '读取流失败：' + (e3 && e3.message ? e3.message : e3));
      }
      if (chunk.done) break;
      buf += dec.decode(chunk.value, { stream: true });
      var idx;
      while ((idx = buf.indexOf('\n\n')) >= 0) {
        var frame = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        var lines = frame.split('\n');
        for (var i = 0; i < lines.length; i++) {
          var line = lines[i];
          if (line.indexOf('data:') !== 0) continue;
          var body = line.slice(5).trim();
          if (body === '[DONE]') return;
          var obj;
          try { obj = JSON.parse(body); } catch (e4) { continue; }
          if (kind === 'messages') {
            /* Anthropic fails in-band the same way: a named `error` event. */
            if (obj && obj.type === 'error') {
              try { ctrl.abort(); } catch (e5) {}
              throw new ApiError(0, '生成失败：' +
                ((obj.error && obj.error.message) || 'unknown error'));
            }
            var mchunk = msgEventToChat(obj, mstate);
            if (mchunk) onEvent({ kind: 'sse', data: mchunk });
            continue;
          }
          if (kind !== 'responses') { onEvent({ kind: 'sse', data: obj }); continue; }
          /* Two ways this dialect can fail after the head is on the wire:
             a bare `error` event, or the terminal `response.failed` whose
             response object carries the reason. There is no [DONE] to fall
             back on, so an unnoticed one would leave the bubble hanging. */
          if (obj && obj.type === 'error') {
            try { ctrl.abort(); } catch (e5) {}
            throw new ApiError(0, '生成失败：' + (obj.message || 'unknown error'));
          }
          if (obj && obj.type === 'response.failed') {
            try { ctrl.abort(); } catch (e5) {}
            var rerr = obj.response && obj.response.error;
            throw new ApiError(0, '生成失败：' +
              ((rerr && rerr.message) || 'unknown error'));
          }
          var xlated = respEventToChat(obj);
          if (xlated) onEvent({ kind: 'sse', data: xlated });
        }
      }
    }
  }

  /* A Responses request is a flat item list, not role-keyed turns: an
     assistant turn with two calls is one message item plus two
     function_call items, and a tool result names the call it answers. */
  function respInput(session) {
    var input = [], instructions = [];
    session.messages.forEach(function (m) {
      if (m.role === 'system') { if (m.content) instructions.push(m.content); return; }
      if (m.role === 'tool') {
        input.push({ type: 'function_call_output', call_id: m.toolCallId, output: m.content || '' });
        return;
      }
      if (m.role === 'assistant') {
        if (!m.content && !(m.toolCalls && m.toolCalls.length)) return;
        if (m.content) {
          input.push({ type: 'message', role: 'assistant',
                       content: [{ type: 'output_text', text: m.content }] });
        }
        (m.toolCalls || []).forEach(function (tc, i) {
          input.push({ type: 'function_call', call_id: tc.id || ('call_' + i),
                       name: tc.name, arguments: tc.arguments || '{}' });
        });
        return;
      }
      var content = [];
      if (m.content) content.push({ type: 'input_text', text: m.content });
      (m.images || []).forEach(function (image) {
        var data = imagePayload(image);
        content.push({ type: 'input_image', image_url: data.url });
      });
      input.push({ type: 'message', role: m.role, content: content });
    });
    return { input: input, instructions: instructions.join('\n\n') };
  }
  /* Responses declares tools flat; the console keeps them in the chat form
     because that is what the other dialect wants. */
  function respToolDefs(defs) {
    return defs.map(function (d) {
      var f = d.function || d;
      var out = { type: 'function', name: f.name };
      if (f.description) out.description = f.description;
      if (f.parameters) out.parameters = f.parameters;
      return out;
    });
  }
  function responsesPayload(session, maxOut) {
    var conv = respInput(session);
    var p = {
      model: apiModel(),
      input: conv.input,
      stream: !!state.prefs.stream,
      max_output_tokens: maxOut,
      reasoning: { effort: state.prefs.effort }
    };
    if (conv.instructions) p.instructions = conv.instructions;
    /* There is no separate summariser: asked for a summary the server
       reports the checkpoint's own reasoning block, which is what the
       console's thinking panel shows anyway. */
    if (state.prefs.effort !== 'off') p.reasoning.summary = 'auto';
    if (state.prefs.tools) {
      p.tools = respToolDefs(TOOL_DEFS);
      p.tool_choice = 'auto';
    }
    return p;
  }

  /* ---- the Anthropic request --------------------------------------- */
  /* The console's own effort names map onto the budget ladder this engine
     publishes; `off` is the protocol's own way of saying "answer first". */
  function msgThinking() {
    var budget = { low: 512, medium: 2048, high: 8192, ultra: 16384, extreme: 32768 };
    if (state.prefs.effort === 'off') return { type: 'disabled' };
    return { type: 'enabled', budget_tokens: budget[state.prefs.effort] || 32768 };
  }
  function msgToolDefs(defs) {
    return defs.map(function (d) {
      var f = d.function || d;
      var out = { name: f.name };
      if (f.description) out.description = f.description;
      out.input_schema = f.parameters || { type: 'object', properties: {} };
      return out;
    });
  }
  /* A tool result is a block *inside* the next user turn, not its own role,
     so consecutive results are merged into one message the way the protocol
     expects a batch of parallel calls to come back. */
  function msgMessages(session) {
    var out = [], system = [];
    var lastWasResults = false;
    session.messages.forEach(function (m) {
      if (m.role === 'system') { if (m.content) system.push(m.content); return; }
      if (m.role === 'tool') {
        var block = {
          type: 'tool_result', tool_use_id: m.toolCallId,
          content: m.content || ''
        };
        if (lastWasResults && out.length) out[out.length - 1].content.push(block);
        else out.push({ role: 'user', content: [block] });
        lastWasResults = true;
        return;
      }
      lastWasResults = false;
      if (m.role === 'assistant') {
        if (!m.content && !(m.toolCalls && m.toolCalls.length)) return;
        var blocks = [];
        if (m.content) blocks.push({ type: 'text', text: m.content });
        (m.toolCalls || []).forEach(function (tc, i) {
          var input = {};
          try { input = JSON.parse(tc.arguments || '{}') || {}; } catch (e) { input = {}; }
          blocks.push({
            type: 'tool_use', id: tc.id || ('call_' + i),
            name: tc.name, input: input
          });
        });
        out.push({ role: 'assistant', content: blocks });
        return;
      }
      var blocks = [];
      if (m.content) blocks.push({ type: 'text', text: m.content });
      (m.images || []).forEach(function (image) {
        var data = imagePayload(image);
        blocks.push({ type: 'image', source: {
          type: 'base64', media_type: data.mime, data: data.base64
        } });
      });
      out.push({ role: 'user', content: blocks });
    });
    return { messages: out, system: system.join('\n\n') };
  }
  function messagesPayload(session, maxOut) {
    var conv = msgMessages(session);
    var p = {
      model: apiModel(),
      max_tokens: maxOut,
      stream: !!state.prefs.stream,
      messages: conv.messages,
      thinking: msgThinking()
    };
    if (conv.system) p.system = conv.system;
    if (state.prefs.tools) {
      p.tools = msgToolDefs(TOOL_DEFS);
      p.tool_choice = { type: 'auto' };
    }
    return p;
  }

  function apiMessages(s) {
    var out = [];
    s.messages.forEach(function (m) {
      if (m.role === 'tool') {
        out.push({ role: 'tool', tool_call_id: m.toolCallId, content: m.content });
        return;
      }
      if (m.role === 'assistant') {
        if (!m.content && !(m.toolCalls && m.toolCalls.length)) return;
        var am = { role: 'assistant', content: m.content || '' };
        if (m.toolCalls && m.toolCalls.length) {
          am.tool_calls = m.toolCalls.map(function (tc, i) {
            return {
              id: tc.id || ('call_' + i),
              type: 'function',
              function: { name: tc.name, arguments: tc.arguments || '{}' }
            };
          });
        }
        out.push(am);
        return;
      }
      if (Array.isArray(m.images) && m.images.length) {
        var parts = [];
        if (m.content) parts.push({ type: 'text', text: m.content });
        m.images.forEach(function (image) {
          parts.push({ type: 'image_url', image_url: { url: imagePayload(image).url } });
        });
        out.push({ role: m.role, content: parts });
        return;
      }
      out.push({ role: m.role, content: m.content });
    });
    return out;
  }

  function systemOnePayload(session) {
    var last = null;
    for (var i = session.messages.length - 1; i >= 0; i--) {
      if (session.messages[i].role === 'user') { last = session.messages[i]; break; }
    }
    var options = ($('systemOptions').value || '').split(',').map(function (s) {
      return s.trim();
    }).filter(Boolean);
    if (options.length < 2 || options.length > 26)
      throw new ApiError(0, 'System One needs 2..26 comma-separated options');
    var criteria = {};
    options.forEach(function (option) { criteria[option] = option; });
    var payload = {
      model: apiModel(),
      state: last ? (last.content || (last.images && last.images.length ? 'Image input.' : '')) : '',
      stream: false,
      questions: { decision: {
        type: 'choice',
        instructions: $('systemQuestion').value || 'Which option best describes the input?',
        criteria: criteria
      } }
    };
    if (last && last.images && last.images.length)
      payload.images = last.images.map(function (image) { return imagePayload(image).url; });
    return payload;
  }

  /* ------------------------------------------------------------------ */
  /* one generation                                                      */
  /* ------------------------------------------------------------------ */
  async function generate(session, assistantMsg, streamEl) {
    /* History is replayed on every turn. Resolve every referenced blob before
       building any dialect so an image is never silently omitted. */
    await hydrateSessionImages(session);
    var maxOut = Math.max(16, Number(state.prefs.maxTokens) || 2048);
    var payload;
    if (apiKind() === 'systemone') {
      payload = systemOnePayload(session);
    } else if (apiKind() === 'responses') {
      payload = responsesPayload(session, maxOut);
    } else if (apiKind() === 'messages') {
      payload = messagesPayload(session, maxOut);
    } else {
      payload = {
        messages: apiMessages(session),
        stream: !!state.prefs.stream,
        max_tokens: maxOut,
        reasoning_effort: state.prefs.effort
      };
      payload.model = apiModel();
      /* Chat needs to be told to send the usage frame; Responses always ends
         with `response.completed`, which carries it. */
      if (payload.stream) payload.stream_options = { include_usage: true };
      if (state.prefs.tools) {
        payload.tools = TOOL_DEFS;
        payload.tool_choice = 'auto';
      }
    }

    var t0 = nowMs(), tFirst = 0, tLast = 0;
    var usage = null, finish = null, pending = false, raf = 0;
    var contentEl = streamEl.querySelector('[data-role="content"]');
    var statsEl = streamEl.querySelector('[data-role="stats"]');
    var thinkEl = streamEl.querySelector('[data-role="think"]');
    var thinkBox = streamEl.querySelector('.think');
    var thinkLabel = thinkBox ? thinkBox.querySelector('.tag') : null;
    var thinkTitle = thinkBox ? thinkBox.querySelector('.summary-title') : null;
    var toolsEl = streamEl.querySelector('[data-role="tools"]');
    var collapsedThink = false;

    /* Same reason as the reasoning panel: a call container is only created
       once there is something to put in it. It sits above the prose so the
       three live regions read in the order they were produced. */
    function ensureTools() {
      if (toolsEl || !contentEl) return;
      if (!assistantMsg.toolStream && !(assistantMsg.toolCalls && assistantMsg.toolCalls.length)) return;
      toolsEl = document.createElement('div');
      toolsEl.className = 'tools';
      toolsEl.setAttribute('data-role', 'tools');
      contentEl.parentNode.insertBefore(toolsEl, contentEl);
    }

    /* The reasoning panel does not exist until the model actually thinks:
       creating it up front would leave an empty box on every direct answer. */
    function ensureThink() {
      if (thinkBox || !assistantMsg.reasoning || !contentEl) return;
      thinkBox = document.createElement('details');
      thinkBox.className = 'think';
      thinkBox.open = true;
      thinkBox.innerHTML = '<summary><span class="chev">▶</span>' +
        '<span class="summary-title">正在思考…</span>' +
        '<span class="tag">0 字</span></summary>' +
        '<div class="think-body" data-role="think"></div>';
      /* above the tool container, so the three regions stay in the order the
         model produced them even when a call was registered first */
      contentEl.parentNode.insertBefore(thinkBox, toolsEl || contentEl);
      thinkEl = thinkBox.querySelector('[data-role="think"]');
      thinkLabel = thinkBox.querySelector('.tag');
      thinkTitle = thinkBox.querySelector('.summary-title');
    }

    function paint() {
      pending = false;
      if (!contentEl) return;
      ensureThink();
      ensureTools();
      /* All three live regions are painted from the same snapshot: reasoning
         in its panel, the tool call (in flight, or parsed once it closes),
         and the prose. None of them waits for the others to finish. */
      if (toolsEl) {
        toolsEl.innerHTML = (assistantMsg.toolStream ? toolStreamHtml() : '') +
          toolCallItems(assistantMsg.toolCalls);
        var tsBody = toolsEl.querySelector('.ts-body');
        if (tsBody) {
          tsBody.textContent = assistantMsg.toolStream;
          tsBody.scrollTop = tsBody.scrollHeight;
        }
      }
      contentEl.innerHTML =
        (assistantMsg.content ? renderMd(assistantMsg.content) : '') +
        '<span class="caret"></span>';
      if (thinkEl) thinkEl.textContent = assistantMsg.reasoning;
      if (thinkTitle) thinkTitle.textContent = assistantMsg.content ? '思考过程' : '正在思考…';
      if (thinkLabel) thinkLabel.textContent = fmtNum(assistantMsg.reasoning.length) + ' 字';
      if (thinkBox && assistantMsg.content && !collapsedThink) {
        thinkBox.open = false;
        collapsedThink = true;
      }
      if (statsEl) statsEl.textContent = liveStats();
      scrollDown(false);
    }
    function schedule() {
      if (pending) return;
      pending = true;
      raf = requestAnimationFrame(paint);
    }
    function liveStats() {
      var el = tLast ? (tLast - (tFirst || t0)) / 1000 : 0;
      var toks = usage && usage.completion_tokens ? usage.completion_tokens : null;
      var parts = [];
      if (tFirst) parts.push('TTFT ' + ((tFirst - t0) / 1000).toFixed(2) + 's');
      if (el > 0.05) parts.push((toks ? (toks / el) : 0).toFixed(2) + ' tok/s');
      if (toks) parts.push(toks + ' tok');
      if (assistantMsg.toolStream) parts.push('工具调用 ' + assistantMsg.toolStream.length + ' 字符');
      if (assistantMsg.reasoning) parts.push('思考 ' + assistantMsg.reasoning.length + ' 字');
      return parts.join(' · ');
    }

    function onEvent(ev) {
      if (ev.kind === 'full') {
        var choice = (ev.data.choices || [])[0] || {};
        var msg = choice.message || {};
        assistantMsg.content = msg.content || '';
        assistantMsg.reasoning = msg.reasoning_content || '';
        assistantMsg.toolCalls = (msg.tool_calls || []).map(function (tc) {
          return { id: tc.id, name: (tc.function || {}).name, arguments: (tc.function || {}).arguments };
        });
        if (ev.data.usage) usage = ev.data.usage;
        finish = choice.finish_reason || null;
        tFirst = tLast = nowMs();
        paint();
        return;
      }
      var data = ev.data;
      if (data.usage) usage = data.usage;
      var ch = (data.choices || [])[0];
      if (!ch) return;
      var d = ch.delta || {};
      if (d.reasoning_content) {
        if (!tFirst) tFirst = nowMs();
        assistantMsg.reasoning += d.reasoning_content;
        schedule();
      }
      if (d.content) {
        if (!tFirst) tFirst = nowMs();
        assistantMsg.content += d.content;
        tLast = nowMs();
        schedule();
      }
      if (d.tool_calls && d.tool_calls.length) {
        /* A call now opens with id/name/arguments:"" and its arguments
           arrive as further chunks sharing the same index. */
        applyToolCallDeltas(assistantMsg, d.tool_calls);
        schedule();
      }
      /* Non-standard progress field (see server.c): the tool-call body as it is
         being generated. Once a standard call card is open, its argument
         deltas are the better progress view, so stop appending the raw XML. */
      if (d.cwick_tool && !assistantMsg.toolStreamMuted) {
        if (!tFirst) tFirst = nowMs();
        assistantMsg.toolStream = (assistantMsg.toolStream || '') + d.cwick_tool;
        tLast = nowMs();
        schedule();
      }
      if (ch.finish_reason) finish = ch.finish_reason;
    }

    await postChat(payload, onEvent);
    if (raf) cancelAnimationFrame(raf);
    tLast = tLast || nowMs();
    var decodeSec = tFirst ? Math.max(0.001, (tLast - tFirst) / 1000) : 0;
    var toks = usage && usage.completion_tokens ? usage.completion_tokens : 0;
    var genTps = decodeSec > 0 && toks > 0 ? toks / decodeSec : 0;
    if (genTps > 0) {
      state.lastTokS = genTps;
      $('livePill').textContent = genTps.toFixed(2) + ' tok/s';
    }
    var stats = [];
    if (tFirst) stats.push('TTFT ' + ((tFirst - t0) / 1000).toFixed(2) + 's');
    if (genTps) stats.push(genTps.toFixed(2) + ' tok/s');
    if (toks) stats.push(toks + ' tok');
    if (usage && usage.prompt_tokens) stats.push('prompt ' + usage.prompt_tokens + ' tok');
    if (decodeSec) stats.push('decode ' + decodeSec.toFixed(1) + 's');
    assistantMsg.stats = stats.join(' · ');
    assistantMsg.finish = finish;
    return { finish: finish, toolCalls: assistantMsg.toolCalls || [] };
  }

  /* ------------------------------------------------------------------ */
  /* turn loop (send -> optional tool round trips -> final answer)        */
  /* ------------------------------------------------------------------ */
  async function send(text, images) {
    if (state.busy || state.sending) return;
    images = (images || []).map(imageRef).filter(Boolean);
    if (!text.trim() && !images.length) return;

    state.sending = true;
    setBusy(state.busy);
    try {
      await hydrateImages(images);
    } catch (e) {
      restoreDraftImages(images, text);
      setStatus(t('chat.imageFailed') + String(e && e.message ? e.message : e));
      state.sending = false;
      setBusy(false);
      return;
    }

    var s = current() || newSession();
    if (!s.messages.length) {
      s.title = text.trim().slice(0, 28) || (images[0] && images[0].name) || t('chat.imageTitle');
      renderSessions();
    }
    s.messages.push({ role: 'user', content: text, images: images });
    saveSessions();

    var box = $('messages');
    if (box.querySelector('.empty')) box.innerHTML = '';
    box.appendChild(msgNode(s.messages[s.messages.length - 1], false));
    scrollDown(true);

    setBusy(true);
    var startedAt = nowMs();
    var waitingTimer = setInterval(function () {
      var sec = (nowMs() - startedAt) / 1000;
      $('queueText').textContent = '等待首字… ' + sec.toFixed(1) + 's（单槽排队，长提示需要先预填充）';
    }, 200);
    $('queueNote').hidden = false;

    try {
      for (var round = 0; round < MAX_ROUNDS; round++) {
        var am = { role: 'assistant', content: '', reasoning: '', toolCalls: [], stats: '' };
        s.messages.push(am);
        var el = msgNode(am, true);
        box.appendChild(el);
        scrollDown(true);
        state.streaming = { msg: am, el: el };

        var result = await generate(s, am, el);

        /* replace the streaming node with a settled one */
        var settled = msgNode(am, false);
        box.replaceChild(settled, el);
        state.streaming = null;

        var calls = am.toolCalls || [];
        if (state.prefs.tools && calls.length && round < MAX_ROUNDS - 1) {
          for (var i = 0; i < calls.length; i++) {
            var out = runTool(calls[i].name, calls[i].arguments);
            calls[i].result = out;
            s.messages.push({
              role: 'tool',
              toolCallId: calls[i].id || ('call_' + i),
              content: out
            });
          }
          saveSessions();
          var reEl = msgNode(am, false);
          box.replaceChild(reEl, settled);
          scrollDown(true);
          continue;
        }
        break;
      }
      saveSessions();
      setStatus('完成 · 请求用了 ' + ((nowMs() - startedAt) / 1000).toFixed(1) + 's');
    } catch (e) {
      var am2 = s.messages[s.messages.length - 1];
      if (am2 && am2.role === 'assistant') {
        am2.error = (e && e.name === 'AbortError') ? '已停止生成。' : apiErrorText(e);
        var old = state.streaming ? state.streaming.el : null;
        if (old && old.parentNode) old.parentNode.replaceChild(msgNode(am2, false), old);
      }
      state.streaming = null;
      saveSessions();
      setStatus((e && e.name === 'AbortError') ? '已停止' : '请求失败');
    } finally {
      clearInterval(waitingTimer);
      $('queueNote').hidden = true;
      state.sending = false;
      setBusy(false);
      state.abort = null;
      loadMetrics(true);
    }
  }

  function setBusy(b) {
    state.busy = b;
    var locked = b || state.sending || state.imageBusy;
    $('sendBtn').disabled = locked;
    $('sendBtn').hidden = locked;
    $('stopBtn').hidden = !b;
    $('input').disabled = b;
    $('attachBtn').disabled = locked;
    renderDraftImages();
    setHistoryImageControls();
  }

  function stop() {
    if (state.abort) { try { state.abort.abort(); } catch (e) {} }
  }

  /* ------------------------------------------------------------------ */
  /* apps — a manifest-driven H5 market hosted by this server            */
  /* ------------------------------------------------------------------ */
  /* Every app is its own directory under `apps/` and is compiled into the
     binary by the C generator tools/embed_web.c. The market is just a fetch of
     /apps/manifest.json; opening an app points an iframe at its entry, so an
     app owns its DOM/CSS/JS and still calls the same-origin OpenAI endpoint.
     Nothing here needs a second server, a bundler or CORS. */
  var appsState = { loaded: false, list: [], open: false, active: null };

  function appsApiState(ok, text) {
    var el = $('appApiState');
    if (!el) return;
    el.innerHTML = '<i class="dot ' + (ok ? '' : 'bad') + '"></i><span>' +
      esc(text || (ok ? 'API 就绪' : 'API 不可用')) + '</span>';
  }

  async function loadApps(force) {
    if (appsState.loaded && !force) return;
    var grid = $('appGrid');
    if (grid && !appsState.loaded) {
      grid.innerHTML = '<div class="app-loading muted">读取 <code>/apps/manifest.json</code>…</div>';
    }
    try {
      var res = await fetch('/apps/manifest.json', { cache: 'no-store' });
      if (!res.ok) throw new Error('HTTP ' + res.status);
      var j = await res.json();
      appsState.list = Array.isArray(j.apps) ? j.apps : [];
      appsState.loaded = true;
      renderApps();
      setStatus('应用市场：' + appsState.list.length + ' 个应用');
    } catch (e) {
      appsState.loaded = false;
      if (grid) {
        grid.innerHTML = '<div class="app-loading bad">应用清单不可用：' +
          esc(e && e.message ? e.message : e) + '</div>';
      }
    }
  }

  function renderApps() {
    var grid = $('appGrid');
    if (!grid) return;
    if (!appsState.list.length) {
      grid.innerHTML = '<div class="app-loading muted">清单为空：在 apps/manifest.json 中登记应用。</div>';
      return;
    }
    var html = '';
    for (var i = 0; i < appsState.list.length; i++) {
      var a = appsState.list[i] || {};
      html += '<article class="app-card" data-app="' + esc(a.id || '') + '">' +
        '<div class="app-icon">' + esc(a.icon || '?') + '</div>' +
        '<div class="app-body">' +
        '<div class="app-title">' + esc(a.name || a.id || '应用') +
        (a.tag ? ' <span class="tag">' + esc(a.tag) + '</span>' : '') + '</div>' +
        '<p>' + esc(a.description || '') + '</p>' +
        '<div class="app-meta mono">' + esc(a.meta || a.entry || '') + '</div>' +
        '</div>' +
        '<div class="app-actions">' +
        '<span class="mono muted">' + esc(a.version ? 'v' + a.version : '') + '</span>' +
        '<button class="send wide app-open" data-app="' + esc(a.id || '') + '">打开</button>' +
        '</div></article>';
    }
    html += '<article class="app-card soon" aria-disabled="true">' +
      '<div class="app-icon muted">?</div><div class="app-body">' +
      '<div class="app-title">更多应用</div>' +
      '<p>新增一个 H5 应用：创建 <code>apps/&lt;id&gt;/</code>，在 ' +
      '<code>apps/manifest.json</code> 登记，然后 <code>make</code>。' +
      '服务器会把整个目录编译进二进制。</p>' +
      '<div class="app-meta mono">apps/ · manifest.json</div></div>' +
      '<div class="app-actions"><span class="mono muted">预留位</span></div></article>';
    grid.innerHTML = html;
  }

  function openApp(id) {
    var a = null;
    for (var i = 0; i < appsState.list.length; i++) {
      if (appsState.list[i].id === id) { a = appsState.list[i]; break; }
    }
    if (!a) return;
    appsState.open = true;
    appsState.active = a;
    $('appMarket').hidden = true;
    $('appHost').hidden = false;
    $('appHostTitle').textContent = a.name || a.id;
    $('appHostSub').textContent = (a.description || '') + (a.entry ? ' · ' + a.entry : '');
    $('appOpenTab').href = a.entry || '#';
    /* about:blank first: assigning a new src to an iframe already showing an
       app is fine, but resetting when switching apps avoids carrying an old
       page's timers into the new one. */
    $('appFrame').src = 'about:blank';
    var src = a.entry || 'about:blank';
    if (src !== 'about:blank') src += (src.indexOf('?') >= 0 ? '&' : '?') + 'embed=1';
    $('appFrame').src = src;
    setStatus('应用：' + (a.name || a.id));
  }

  function closeApp() {
    appsState.open = false;
    appsState.active = null;
    $('appHost').hidden = true;
    $('appMarket').hidden = false;
    if ($('appFrame')) $('appFrame').src = 'about:blank';
    setStatus('应用市场');
  }

  function appsSyncView() {
    if (!$('appHost')) return;
    $('appMarket').hidden = appsState.open;
    $('appHost').hidden = !appsState.open;
  }

  function appsBind() {
    if ($('appRefresh')) $('appRefresh').addEventListener('click', function () { loadApps(true); });
    if ($('appBack')) $('appBack').addEventListener('click', closeApp);
    if ($('appGrid')) $('appGrid').addEventListener('click', function (e) {
      var card = e.target.closest('.app-card');
      if (!card || !card.dataset.app) return;
      openApp(card.dataset.app);
    });
  }

  /* ------------------------------------------------------------------ */
  /* metric trends                                                       */
  /* ------------------------------------------------------------------ */
  /* /v1/metrics publishes cumulative counters. Plotting them verbatim draws
     a lifetime average, which is not a trend: each point is the delta between
     two scrapes. A window with no decode steps gets a null, drawn as a break
     in the line rather than a false zero. Four canvases, no chart library. */
  var TREND_MAX = 120;                    /* 2 s x 120 = 4 minutes */
  var trend = {
    at: [], tokS: [], mem: [], hit: [], pfx: [], ttft: [],
    prev: null, lastAt: 0
  };

  function trendSample(s) {
    var toggle = $('liveToggle');
    if (toggle && !toggle.checked) return;      /* paused: hold the plot */
    if (trend.lastAt && s.at - trend.lastAt < 900) return;
    var p = trend.prev;
    trend.prev = s;
    trend.lastAt = s.at;
    if (!p) return;                             /* need two scrapes for a delta */

    var ds = s.steps - p.steps, dt = s.stepSum - p.stepSum;
    var dn = s.ttftN - p.ttftN, dl = s.ttftSum - p.ttftSum;
    var dh = s.hit - p.hit, dm = s.miss - p.miss;
    var dph = s.ph - p.ph, dpq = s.pq - p.pq;

    trend.tokS.push(ds > 0 && dt > 0 ? ds / dt : null);
    trend.ttft.push(dn > 0 && dl > 0 ? dl / dn : null);
    trend.hit.push(dh + dm > 0 ? 100 * dh / (dh + dm) : null);
    trend.pfx.push(dpq > 0 ? 100 * dph / dpq : null);
    trend.mem.push(s.foot / 1073741824);
    trend.at.push(s.at);
    var keys = ['at', 'tokS', 'mem', 'hit', 'pfx', 'ttft'];
    for (var i = 0; i < keys.length; i++) {
      if (trend[keys[i]].length > TREND_MAX) trend[keys[i]].shift();
    }
    pmSample(s.all || {}, s.at, p.at ? (s.at - p.at) / 1000 : 0);
    trendDrawAll();
  }

  function trendClear() {
    trend.at = []; trend.tokS = []; trend.mem = [];
    trend.hit = []; trend.pfx = []; trend.ttft = [];
    trend.prev = null; trend.lastAt = 0;
    pmResetHist();
    trendDrawAll();
  }

  function trendLast(arr) {
    for (var i = arr.length - 1; i >= 0; i--) {
      if (arr[i] != null && isFinite(arr[i])) return arr[i];
    }
    return null;
  }

  /* A hidden section has clientWidth 0, so fall back to the width attribute
     from the markup; the canvas still has a bitmap to draw into. showView()
     and resize redraw once it becomes visible and gets its real width. */
  function chartCanvas(id) {
    var c = $(id);
    if (!c || !c.getContext) return null;
    var w = c.clientWidth || 600, h = 130;
    var dpr = window.devicePixelRatio || 1;
    c.width = Math.round(w * dpr);
    c.height = Math.round(h * dpr);
    c.style.height = h + 'px';
    var ctx = c.getContext('2d');
    if (!ctx) return null;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    return { ctx: ctx, w: w, h: h };
  }

  function drawChart(id, axisId, series, opt) {
    var c = chartCanvas(id);
    if (!c) return;
    var ctx = c.ctx, w = c.w, h = c.h;
    var L = 42, R = 8, T = 10, B = 10;
    var pw = Math.max(1, w - L - R), ph = Math.max(1, h - T - B);
    ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.lineWidth = 1;
    ctx.strokeStyle = '#1e2634';
    for (var g = 0; g <= 2; g++) {
      var gy = T + ph * g / 2 + 0.5;
      ctx.beginPath(); ctx.moveTo(L, gy); ctx.lineTo(w - R, gy); ctx.stroke();
    }

    var n = series[0].data.length;
    var axis = $(axisId);
    /* A custom series may be younger than the scrape clock, so measure its
       own window rather than the whole session's first sample. */
    var times = opt.times || trend.at;
    var t0 = times.length > 1 ? times[Math.max(0, times.length - n)] : 0;
    var t1 = times.length ? times[times.length - 1] : 0;
    var spanSec = t1 > t0 ? (t1 - t0) / 1000 : 0;
    var empty = opt.empty || '等待生成活动…';
    if (axis) {
      axis.innerHTML = '<span>−' + (spanSec > 0 ? fmtSec(spanSec) : '0 s') +
        '</span> <span>' +
        (opt.legend ? opt.legend + ' · ' : '') + 'now · ' + n + ' 点</span>';
    }
    if (!n) {
      ctx.fillStyle = '#66728a';
      ctx.textAlign = 'left';
      ctx.fillText('等待采样…', L + 8, T + ph / 2);
      return;
    }

    var lo = opt.min != null ? opt.min : Infinity;
    var hi = opt.max != null ? opt.max : -Infinity;
    var any = false;
    for (var si = 0; si < series.length; si++) {
      var d = series[si].data;
      for (var k = 0; k < d.length; k++) {
        var v = d[k];
        if (v == null || !isFinite(v)) continue;
        any = true;
        if (opt.min == null && v < lo) lo = v;
        if (opt.max == null && v > hi) hi = v;
      }
    }
    if (!any) {
      ctx.fillStyle = '#66728a';
      ctx.textAlign = 'left';
      ctx.fillText(empty, L + 8, T + ph / 2);
      return;
    }
    var span = hi - lo;
    /* A flat series still needs a visible band. Scaling it by the value keeps
       a 0.01 s TTFT from being flattened against a 1 s axis. */
    if (!(span > 0)) span = Math.max(Math.abs(hi) * 0.1, 1e-6);
    var pad = span * 0.12;
    var yLo = opt.min != null ? lo : Math.max(0, lo - pad);
    var yHi = hi + pad;
    var range = yHi - yLo || 1;

    ctx.fillStyle = '#66728a';
    ctx.textAlign = 'left';
    ctx.fillText(opt.fmt(yHi), 3, T + 8);
    ctx.fillText(opt.fmt(yLo), 3, T + ph - 1);

    for (var q = 0; q < series.length; q++) {
      var data = series[q].data;
      ctx.strokeStyle = series[q].color;
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      var started = false, lastX = 0, lastY = 0;
      for (var i = 0; i < data.length; i++) {
        var yv = data[i];
        if (yv == null || !isFinite(yv)) { started = false; continue; }
        var x = L + (data.length <= 1 ? pw : pw * i / (data.length - 1));
        var y = T + ph - ((yv - yLo) / range) * ph;
        y = Math.max(T, Math.min(T + ph, y));
        if (!started) { ctx.moveTo(x, y); started = true; }
        else ctx.lineTo(x, y);
        lastX = x; lastY = y;
      }
      ctx.stroke();
      if (data.length && data[data.length - 1] != null &&
          isFinite(data[data.length - 1])) {
        ctx.fillStyle = series[q].color;
        ctx.beginPath(); ctx.arc(lastX, lastY, 2.6, 0, Math.PI * 2); ctx.fill();
      }
    }
  }

  function trendDrawAll() {
    var n = trend.at.length;
    if ($('trendSpan')) $('trendSpan').textContent = n
      ? n + ' 点 · ' + fmtSec(n > 1 ? (trend.at[n - 1] - trend.at[0]) / 1000 : 0)
      : '采集中…';
    var v = trendLast(trend.tokS);
    $('trendTokS').textContent = v == null ? '—' : v.toFixed(2);
    v = trendLast(trend.mem);
    $('trendMem').textContent = v == null ? '—' : v.toFixed(2);
    var h = trendLast(trend.hit), pf = trendLast(trend.pfx);
    $('trendHit').textContent = h == null && pf == null ? '—'
      : (h == null ? '—' : h.toFixed(0) + '%') + ' / ' +
        (pf == null ? '—' : pf.toFixed(0) + '%');
    v = trendLast(trend.ttft);
    $('trendTtft').textContent = v == null ? '—' : v.toFixed(2);

    var f2 = function (x) { return x.toFixed(2); };
    var f1 = function (x) { return x.toFixed(1); };
    var f0 = function (x) { return x.toFixed(0); };
    var fT = function (x) { return x < 1 ? x.toFixed(3) : x.toFixed(2); };
    drawChart('chartTokS', 'axisTokS',
              [{ data: trend.tokS, color: '#6f8cff' }], { fmt: f2 });
    drawChart('chartMem', 'axisMem',
              [{ data: trend.mem, color: '#3ddc97' }], { fmt: f1 });
    drawChart('chartHit', 'axisHit',
              [{ data: trend.hit, color: '#6f8cff' },
               { data: trend.pfx, color: '#9b7bff' }],
              { min: 0, max: 100, fmt: f0, legend: '专家蓝 / 前缀紫' });
    drawChart('chartTtft', 'axisTtft',
              [{ data: trend.ttft, color: '#f5b95c' }], { fmt: fT });

    var note = $('trendNote');
    if (note) {
      note.hidden = false;
      note.innerHTML = 'Prometheus 指标：<code>cwick:decode_step_seconds</code> · ' +
        '<code>cwick:process_footprint_bytes</code> · ' +
        '<code>cwick:expert_cache_hits_total</code> / ' +
        '<code>vllm:prefix_cache_hits_total</code> · ' +
        '<code>vllm:time_to_first_token_seconds</code>。<br>' +
        '速率按相邻采样的增量计算；空闲窗口（没有新的解码步）不落点，折线会断开。' +
        '关闭「实时刷新」会暂停采样，rail 上的常驻内存仍每 5 秒更新。';
    }
  }

  /* model_info carries more labels than the six headline cells above; show
     all of them so a model switch or a different checkpoint family is visible
     without opening the Prometheus endpoint. */
  function renderModelLabels(info) {
    var box = $('modelInfoAll');
    if (!box) return;
    var keys = Object.keys(info || {});
    if (!keys.length) {
      box.innerHTML = '<span class="muted">cwick:model_info 未提供标签</span>';
      return;
    }
    keys.sort();
    box.innerHTML = keys.map(function (k) {
      return '<span class="lb"><i>' + esc(k) + '</i> = <b>' +
        esc(info[k]) + '</b></span>';
    }).join('');
  }

  /* ------------------------------------------------------------------ */
  /* Prometheus series browser                                          */
  /* ------------------------------------------------------------------ */
  /* The four charts above are a fixed projection of the exporter. This
     panel exposes the exporter itself: every series, its HELP/TYPE and its
     current value. Picking one attaches a private canvas with its own
     y-axis, so a byte counter and a ratio never share a scale. Counters can
     be plotted as per-second deltas, which is usually what "trend" means
     for them. The selection is a browser preference, not a server setting. */
  var PM_KEY = 'cwick.prom.v1';
  var PM_MAX = 4;
  var PM_COLORS = ['#6f8cff', '#3ddc97', '#f5b95c', '#9b7bff'];
  var pm = {
    keys: [], sig: '', values: {}, help: {}, type: {},
    selected: [], hist: {}, prev: {}, at: [],
    rate: false, query: ''
  };

  function pmLoad() {
    try {
      var p = JSON.parse(localStorage.getItem(PM_KEY) || '{}') || {};
      if (Array.isArray(p.selected)) {
        pm.selected = p.selected.filter(function (x) {
          return typeof x === 'string' && x.length < 200;
        }).slice(0, PM_MAX);
      }
      pm.rate = !!p.rate;
    } catch (e) { /* private mode: the selection is ephemeral */ }
  }
  function pmSave() {
    try {
      localStorage.setItem(PM_KEY,
        JSON.stringify({ selected: pm.selected, rate: pm.rate }));
    } catch (e) {}
  }
  function pmFmt(v) {
    if (v == null || !isFinite(v)) return '—';
    var a = Math.abs(v);
    if (a >= 1e9) return (v / 1e9).toFixed(2) + 'G';
    if (a >= 1e6) return (v / 1e6).toFixed(2) + 'M';
    if (a >= 1e4) return (v / 1e3).toFixed(1) + 'K';
    if (a >= 100) return v.toFixed(0);
    if (a >= 1) return v.toFixed(2);
    if (a >= 1e-3) return v.toFixed(4);
    return v.toExponential(1);
  }
  /* Prometheus rate() applies to counters. Differentiating a gauge yields a
     constant slope (uptime becomes ~1) that looks like a measurement but is
     not one. TYPE decides; the usual suffixes cover exporters that omit the
     comment line. */
  function pmIsCounter(name) {
    var t = pm.type[name];
    if (t) return t === 'counter';
    return /_(total|count|sum)$/.test(name);
  }
  function pmSyncSelNote() {
    var el = $('pmSelNote');
    if (el) el.textContent = pm.selected.length
      ? '已选 ' + pm.selected.length + '/' + PM_MAX + ' · 点行可移除'
      : '未选择指标（最多 ' + PM_MAX + ' 个）';
  }

  function pmIngest(m) {
    var keys = Object.keys(m).filter(function (k) {
      return k.charAt(0) !== '_';
    }).sort();
    pm.values = m;
    pm.help = m.__help || pm.help;
    pm.type = m.__type || pm.type;
    var sig = keys.join('|');
    if (sig !== pm.sig) { pm.sig = sig; pm.keys = keys; pmRenderList(); }
    else pmUpdateValues();
    $('pmCount').textContent = keys.length + ' 条序列';
  }

  function pmUpdateValues() {
    var nodes = document.querySelectorAll('#pmList [data-pm-val]');
    for (var i = 0; i < nodes.length; i++) {
      var v = pm.values[nodes[i].dataset.pmVal];
      nodes[i].textContent = (typeof v === 'number' && isFinite(v)) ? pmFmt(v) : '—';
    }
  }

  function pmRenderList() {
    var box = $('pmList');
    if (!box) return;
    var q = pm.query.toLowerCase();
    var hits = pm.keys.filter(function (k) {
      return !q || k.toLowerCase().indexOf(q) >= 0;
    });
    if (!hits.length) {
      box.innerHTML = '<div class="muted">没有匹配的指标。</div>';
      pmSyncSelNote();
      return;
    }
    box.innerHTML = hits.map(function (k) {
      var on = pm.selected.indexOf(k) >= 0;
      var v = pm.values[k];
      var type = pm.type[k] || '', help = pm.help[k] || '';
      return '<div class="pm-row' + (on ? ' is-on' : '') + '" data-pm="' + esc(k) + '">' +
        '<span class="pm-name" title="' + esc(k) + '">' + esc(k) + '</span>' +
        '<span class="pm-meta" title="' + esc(help) + '">' +
          (type ? '<em>' + esc(type) + '</em>' : '') + esc(help) + '</span>' +
        '<b class="pm-val" data-pm-val="' + esc(k) + '">' +
          (typeof v === 'number' && isFinite(v) ? pmFmt(v) : '—') + '</b>' +
        '<button class="min-btn' + (on ? ' is-on' : '') + '" type="button" data-pm="' +
          esc(k) + '">' + (on ? '移除' : '+ 趋势') + '</button></div>';
    }).join('');
    pmSyncSelNote();
  }

  function pmToggle(name) {
    var i = pm.selected.indexOf(name);
    if (i >= 0) {
      pm.selected.splice(i, 1);
      delete pm.hist[name]; delete pm.prev[name];
    } else {
      if (pm.selected.length >= PM_MAX) { pmSyncSelNote(); return; }
      pm.selected.push(name);
      pm.hist[name] = [];
      delete pm.prev[name];
    }
    pmSave();
    pmRenderList();
    pmRenderCharts();
  }

  function pmResetHist() {
    pm.hist = {}; pm.prev = {}; pm.at = [];
    for (var i = 0; i < pm.selected.length; i++) pm.hist[pm.selected[i]] = [];
    pmDrawAll();
  }

  function pmRenderCharts() {
    var box = $('pmCharts');
    if (!box) return;
    if (!pm.selected.length) {
      box.innerHTML = '<div class="pm-empty muted">从下方列表点选指标，' +
        '即可为它单独画一条趋势线（最多 4 个）。</div>';
      return;
    }
    box.innerHTML = pm.selected.map(function (name, i) {
      return '<div class="trend-cell">' +
        '<div class="trend-title"><span title="' + esc(name) + '">' + esc(name) +
          '</span><b class="mono" id="pmVal' + i + '">—</b></div>' +
        '<canvas id="pmChart' + i + '" width="600" height="130"></canvas>' +
        '<div class="trend-axis" id="pmAxis' + i + '">—</div></div>';
    }).join('');
    pmDrawAll();
  }

  function pmDrawAll() {
    for (var i = 0; i < pm.selected.length; i++) {
      var name = pm.selected[i];
      var data = pm.hist[name] || [];
      var v = trendLast(data);
      var el = $('pmVal' + i);
      if (el) el.textContent = v == null ? '—' : pmFmt(v);
      drawChart('pmChart' + i, 'pmAxis' + i,
                [{ data: data, color: PM_COLORS[i % PM_COLORS.length] }],
                { fmt: pmFmt, times: pm.at, empty: '等待采样…',
                  legend: (pm.rate && pmIsCounter(name)) ? '每秒增量' : '原始值' });
    }
  }

  function pmSample(m, at, dt) {
    if (!pm.selected.length) return;
    pm.at.push(at);
    if (pm.at.length > TREND_MAX) pm.at.shift();
    for (var i = 0; i < pm.selected.length; i++) {
      var name = pm.selected[i];
      var cur = (typeof m[name] === 'number' && isFinite(m[name])) ? m[name] : null;
      var val = null;
      if (cur != null) {
        if (pm.rate && pmIsCounter(name)) {
          var prev = pm.prev[name];
          val = (prev != null && dt > 0) ? (cur - prev) / dt : null;
        } else {
          val = cur;
        }
      }
      pm.prev[name] = cur;
      if (!pm.hist[name]) pm.hist[name] = [];
      pm.hist[name].push(val);
      if (pm.hist[name].length > TREND_MAX) pm.hist[name].shift();
    }
    pmDrawAll();
  }

  /* ------------------------------------------------------------------ */
  /* metrics                                                             */
  /* ------------------------------------------------------------------ */
  function parseMetrics(text) {
    var out = {};
    var lines = text.split('\n');
    for (var i = 0; i < lines.length; i++) {
      var line = lines[i];
      if (!line) continue;
      if (line.charAt(0) === '#') {
        var hm = line.match(/^# HELP (\S+)\s+(.*)$/);
        if (hm) { (out.__help || (out.__help = {}))[hm[1]] = hm[2]; continue; }
        var tm = line.match(/^# TYPE (\S+)\s+(\S+)/);
        if (tm) { (out.__type || (out.__type = {}))[tm[1]] = tm[2]; continue; }
        continue;
      }
      var sp = line.lastIndexOf(' ');
      if (sp <= 0) continue;
      var key = line.slice(0, sp);
      var val = parseFloat(line.slice(sp + 1));
      if (!isFinite(val)) continue;
      var brace = key.indexOf('{');
      var name = brace >= 0 ? key.slice(0, brace) : key;
      if (!(name in out)) out[name] = val;   /* first sample wins */
    }
    var mi = text.match(/cwick:model_info\{([^}]*)\}/);
    if (mi) {
      var labels = {};
      mi[1].replace(/(\w+)="([^"]*)"/g, function (m, k, v) { labels[k] = v; return m; });
      out.__model = labels;
    }
    return out;
  }

  function ratio(a, b) {
    if (!b || b <= 0) return null;
    return Math.max(0, Math.min(1, a / b));
  }

  /* The ceiling comes from the memory plan; a synthetic test model has none,
     and "38.8 MiB / —" reads like a bug rather than "no ceiling published". */
  function memText(foot, ceil) {
    if (!ceil) return fmtBytes(foot);
    return fmtBytes(foot) + ' / ' + fmtBytes(ceil) +
      '  (' + ((foot / ceil) * 100).toFixed(0) + '%)';
  }

  function renderMetrics(m) {
    var steps = m['cwick:decode_step_seconds_count'] || 0;
    var stepSum = m['cwick:decode_step_seconds_sum'] || 0;
    var stepMean = steps > 0 ? stepSum / steps : 0;
    var tokS = stepMean > 0 ? 1 / stepMean : 0;

    $('mTokS').innerHTML = tokS > 0 ? tokS.toFixed(2) + '<small>tok/s</small>' : '—';
    $('mTokSFoot').textContent = stepMean > 0
      ? '每步 ' + (stepMean * 1000).toFixed(0) + ' ms · ' + fmtNum(steps) + ' 步'
      : '来自 cwick:decode_step_seconds';

    var ttftN = m['vllm:time_to_first_token_seconds_count'] || 0;
    var ttftSum = m['vllm:time_to_first_token_seconds_sum'] || 0;
    $('mTtft').textContent = ttftN > 0 ? fmtSec(ttftSum / ttftN) : '—';

    var tpotN = m['vllm:time_per_output_token_seconds_count'] || 0;
    var tpotSum = m['vllm:time_per_output_token_seconds_sum'] || 0;
    $('mTpot').textContent = tpotN > 0 ? fmtSec(tpotSum / tpotN) : '—';

    var pt = m['vllm:prompt_tokens_total'] || 0;
    var gt = m['vllm:generation_tokens_total'] || 0;
    $('mTokens').textContent = fmtNum(pt + gt);
    $('mTokensFoot').textContent = 'prompt ' + fmtNum(pt) + ' / gen ' + fmtNum(gt);

    var foot = m['cwick:process_footprint_bytes'] || 0;
    var ceil = m['cwick:memory_ceiling_bytes'] || 0;
    var memRatio = ratio(foot, ceil);
    $('mMemLabel').textContent = memText(foot, ceil);
    $('mMemBar').style.width = (memRatio != null ? memRatio * 100 : 0) + '%';
    $('mMemBar').className = memRatio != null && memRatio > 0.92 ? 'warn' : '';

    var hit = m['cwick:expert_cache_hits_total'] || 0;
    var miss = m['cwick:expert_cache_misses_total'] || 0;
    var hitR = ratio(hit, hit + miss);
    $('mHitLabel').textContent = hitR == null ? '—' : (hitR * 100).toFixed(1) + '%';
    $('mHitBar').style.width = (hitR != null ? hitR * 100 : 0) + '%';

    var ph = m['vllm:prefix_cache_hits_total'] || 0;
    var pq = m['vllm:prefix_cache_queries_total'] || 0;
    var pfxR = ratio(ph, pq);
    $('mPfxLabel').textContent = pfxR == null ? '—' : (pfxR * 100).toFixed(1) + '%  (' +
      fmtNum(ph) + '/' + fmtNum(pq) + ')';
    $('mPfxBar').style.width = (pfxR != null ? pfxR * 100 : 0) + '%';

    var run = m['vllm:num_requests_running'] || 0;
    var wait = m['vllm:num_requests_waiting'] || 0;
    $('mReq').textContent = run + ' / ' + wait;

    var bytes = m['cwick:expert_bytes_streamed_total'] || 0;
    var genT = m['vllm:generation_tokens_total'] || 0;
    $('mStream').textContent = genT > 0 ? fmtBytes(bytes / genT) : '—';
    $('mSteps').textContent = fmtNum(m['cwick:decode_steps_total'] || 0);
    $('mUptime').textContent = fmtSec(m['cwick:uptime_seconds'] || 0);
    $('mSuccess').textContent = fmtNum(m['vllm:request_success_total'] || 0);

    var pd = m['cwick:prefill_tokens_done'] || 0;
    var ptot = m['cwick:prefill_tokens_total'] || 0;
    $('mPrefill').textContent = ptot > 0
      ? fmtNum(pd) + ' / ' + fmtNum(ptot) + ' tok · L' + (m['cwick:prefill_layer'] || 0) +
        '/' + (m['cwick:prefill_layers'] || 0)
      : '空闲';

    var info = m.__model || {};
    if (info.model) {
      $('mModel').textContent = info.model;
      $('brandSub').textContent = info.model;
    }
    if (info.backend) {
      $('mBackend').textContent = info.backend;
      $('backendChip').textContent = 'backend ' + info.backend;
    }
    if (info.spine) $('mSpine').textContent = info.spine;
    if (info.num_layers) $('mLayers').textContent = info.num_layers;
    if (info.num_experts && info.experts_per_token) {
      $('mSparse').textContent = info.experts_per_token + ' / ' + info.num_experts;
    }
    var ctx = m['cwick:context_window_tokens'] || (info.max_context ? Number(info.max_context) : 0);
    if (ctx) {
      $('mCtx').textContent = fmtNum(ctx) + ' tok';
      $('ctxChip').textContent = 'ctx ' + fmtNum(ctx);
    }

    $('memLabel').textContent = memText(foot, ceil);
    $('memBar').style.width = (memRatio != null ? memRatio * 100 : 0) + '%';
    $('memBar').className = memRatio != null && memRatio > 0.92 ? 'warn' : '';
    if (tokS > 0) {
      $('livePill').textContent = tokS.toFixed(2) + ' tok/s';
    }
    $('mUptimePill').textContent = fmtSec(m['cwick:uptime_seconds'] || 0);
    renderModelLabels(info);
    pmIngest(m);
    trendSample({
      at: Date.now(), steps: steps, stepSum: stepSum,
      ttftN: ttftN, ttftSum: ttftSum, hit: hit, miss: miss,
      ph: ph, pq: pq, foot: foot, all: m
    });
  }

  function setConn(ok, text) {
    var pill = $('connPill');
    pill.innerHTML = '<i class="dot ' + (ok ? '' : 'bad') + '"></i><span>' +
      esc(text || (ok ? '已连接' : '未连接')) + '</span>';
  }

  async function loadMetrics(force) {
    try {
      var res = await fetch('/v1/metrics', { cache: 'no-store' });
      if (authRedirect(res)) return;
      if (!res.ok) throw new Error('HTTP ' + res.status);
      var text = await res.text();
      renderMetrics(parseMetrics(text));
      if (force || !state.busy) setConn(true, '已连接');
    } catch (e) {
      setConn(false, '指标不可用');
    }
  }

  async function loadModel() {
    try {
      var res = await fetch('/v1/models', { cache: 'no-store' });
      if (authRedirect(res)) return;
      var j = await res.json();
      var pick = residentOf(j.data || []);
      var id = (pick && pick.id) ? pick.id : 'glm-5.3-flash';
      state.modelId = id;
      $('brandSub').textContent = id;
      $('mModel').textContent = id;
      $('apiBase').textContent = location.origin + '/v1';
      appsApiState(true, 'API 就绪 · ' + id);
      setConn(true, '已连接');
      setStatus('已连接 ' + id);
    } catch (e) {
      appsApiState(false, 'API 不可用');
      setConn(false, '模型不可用');
      setStatus('无法读取 /v1/models');
    }
  }

  async function logout() {
    try { await fetch('/v1/auth/logout', { method: 'POST' }); } catch (e) {}
    location.href = '/login';
  }

  function setStatus(t) { $('statusText').textContent = t; }

  /* Pick the entry the engine actually has resident: /v1/models orders the
     catalog, but after a runtime switch the loaded one is not guaranteed to
     be index 0, and sending the wrong `model` is a 404 on the next request. */
  function residentOf(list) {
    for (var i = 0; i < list.length; i++) {
      if (list[i].cwick && list[i].cwick.loaded) return list[i];
    }
    return list[0] || null;
  }

  /* ------------------------------------------------------------------ */
  /* logs — this process's own stdout / stderr                           */
  /* ------------------------------------------------------------------ */
  /* The server tees fd 1/fd 2 into a bounded ring and publishes it over
     SSE. Two consequences shape this code:
       - a line can be evicted before we read it, so the ring tells us both
         `dropped` (server-side eviction) and `seq`, and we count the hole
         ourselves when reconnecting;
       - watching costs a connection slot out of 64, so the stream is opened
         on entering the view and closed on leaving it, never left running
         behind a tab the user is not looking at. */
  var LS_LOGPREFS = 'cwick.logprefs.v1';
  var LOG_MAX = 3000;          /* lines held in memory and in the DOM */
  var LOG_BACKLOG = 300;       /* history asked for on a cold connect */
  var LOG_CATCHUP = 5000;      /* server caps `limit` at 5000 anyway */

  var logs = {
    lines: [], ctrl: null, busy: false,
    paused: false, follow: true, wrap: false, src: 'all', query: '',
    lastN: 0, expect: 0, gap: 0, dropped: 0, fails: 0,
    timer: 0, pending: [], raf: 0
  };

  function logTime(t) {
    var d = new Date(t);
    var p = function (n, w) { return String(n).padStart(w, '0'); };
    return p(d.getHours(), 2) + ':' + p(d.getMinutes(), 2) + ':' +
           p(d.getSeconds(), 2) + '.' + p(d.getMilliseconds(), 3);
  }

  function logMatches(l) {
    if (logs.src !== 'all' && l.src !== logs.src) return false;
    if (!logs.query) return true;
    return l.msg.toLowerCase().indexOf(logs.query) >= 0;
  }

  /* cwick's own lines start with "cwick: YYYY-MM-DD HH:MM:SS.mmm " — the very
     instant the gutter already shows. Keeping both duplicates the widest
     column on every line and pushes the payload off-screen, so the prefix is
     dropped for *display* only: the gutter takes its time, the raw text stays
     in `logs.lines` (and therefore in the downloaded file) untouched. Lines
     without it — Metal diagnostics, Objective-C output, anything fprintf'd
     straight to stderr — are shown exactly as they arrived. */
  var LOG_PREFIX = /^cwick: \d{4}-\d{2}-\d{2} (\d{2}:\d{2}:\d{2}\.\d{3}) /;

  function logParts(l) {
    var m = LOG_PREFIX.exec(l.msg);
    if (!m) return { t: logTime(l.t), msg: l.msg };
    return { t: m[1], msg: l.msg.slice(m[0].length) };
  }

  function logNode(l) {
    var parts = logParts(l);
    var text = parts.msg;
    var el = document.createElement('div');
    el.className = 'log-line' + (l.src === 'err' ? ' err' : '');
    var html = '<span class="lg-n">' + l.n + '</span>' +
      '<span class="lg-t">' + parts.t + '</span>' +
      '<span class="lg-src">' + (l.src === 'err' ? 'ERR' : 'OUT') + '</span>' +
      '<span class="lg-msg">';
    var q = logs.query;
    if (q) {
      var lower = text.toLowerCase(), from = 0, at;
      while ((at = lower.indexOf(q, from)) >= 0) {
        html += esc(text.slice(from, at)) + '<b class="hit">' +
                esc(text.slice(at, at + q.length)) + '</b>';
        from = at + q.length;
      }
      html += esc(text.slice(from));
    } else {
      html += esc(text);
    }
    el.innerHTML = html + '</span>';
    return el;
  }

  function logRenderAll() {
    var body = $('logBody');
    body.innerHTML = '';
    var frag = document.createDocumentFragment();
    var shown = 0;
    for (var i = 0; i < logs.lines.length; i++) {
      if (!logMatches(logs.lines[i])) continue;
      frag.appendChild(logNode(logs.lines[i]));
      shown++;
    }
    if (!shown) {
      body.innerHTML = '<div class="log-empty">' +
        (logs.lines.length ? '当前过滤条件下没有匹配的行。'
          : '还没有捕获到日志——进程刚启动，或这个实例没有开启日志采集。') +
        '</div>';
      return;
    }
    body.appendChild(frag);
    if (logs.follow) body.scrollTop = body.scrollHeight;
  }

  /* Batched: a 320B model mid-decode prints heartbeats often enough that one
     DOM append per line is measurable, and a firehose with --trace would drop
     frames. One rAF per animation frame is the ceiling that still feels live. */
  function logFlush() {
    logs.raf = 0;
    var batch = logs.pending;
    logs.pending = [];
    if (!batch.length) return;
    var body = $('logBody');
    var empty = body.querySelector('.log-empty');
    if (empty) body.innerHTML = '';
    var atBottom = logs.follow ||
      (body.scrollHeight - body.scrollTop - body.clientHeight < 80);
    var frag = document.createDocumentFragment();
    var added = 0;
    for (var i = 0; i < batch.length; i++) {
      var l = batch[i];
      if (logs.expect > 0 && l.n > logs.expect) logs.gap += l.n - logs.expect;
      logs.expect = l.n + 1;
      logs.lines.push(l);
      if (logs.lines.length > LOG_MAX) logs.lines.shift();
      if (!logMatches(l)) continue;
      frag.appendChild(logNode(l));
      added++;
    }
    if (added) body.appendChild(frag);
    /* The array is capped, so the document has to be too: trim from the top,
       which is the same end the ring itself discards from. */
    while (body.childNodes.length > LOG_MAX) body.removeChild(body.firstChild);
    if (logs.lines.length) logs.lastN = logs.lines[logs.lines.length - 1].n;
    if (atBottom) body.scrollTop = body.scrollHeight;
    logFoot();
  }

  function logPush(l) {
    logs.pending.push(l);
    if (!logs.raf) logs.raf = requestAnimationFrame(logFlush);
  }

  function logFoot() {
    $('logCount').textContent = logs.lines.length + ' 行' +
      (logs.lastN ? ' · 至 #' + logs.lastN : '');
    var d = $('logDrop');
    var bits = [];
    if (logs.dropped) bits.push('服务端丢弃 ' + logs.dropped + ' 行');
    if (logs.gap) bits.push('本次跳过了 ' + logs.gap + ' 行');
    d.hidden = !bits.length;
    d.textContent = bits.join(' · ');
  }

  function logSetState(t) { $('logState').textContent = t; }

  function logFrame(frame) {
    var lines = frame.split('\n');
    for (var i = 0; i < lines.length; i++) {
      var line = lines[i];
      if (!line || line.charAt(0) === ':') continue;      /* keepalive comment */
      if (line.indexOf('data:') !== 0) continue;
      var body = line.slice(5).trim();
      if (!body) continue;
      var o;
      try { o = JSON.parse(body); } catch (e) { continue; }
      if (o.type === 'hello') {
        logs.dropped = o.dropped || 0;
        logFoot();
      } else if (o.type === 'line') {
        logPush({ n: o.n, src: o.src === 'err' ? 'err' : 'out',
                  t: o.t, msg: o.msg || '' });
      }
    }
  }

  function logUrl() {
    /* Cold: let the server choose the recent tail. Warm: resume after our
       last line and ask for the whole catch-up window, so a short disconnect
       (tab in the background, a swap) does not leave a hole. */
    return logs.lastN > 0
      ? '/v1/logs/stream?since=' + (logs.lastN + 1) + '&limit=' + LOG_CATCHUP
      : '/v1/logs/stream?limit=' + LOG_BACKLOG;
  }

  async function openLogStream() {
    if (logs.busy) return;
    logs.busy = true;
    try { await openLogStreamOnce(); } finally { logs.busy = false; }
  }

  async function openLogStreamOnce() {
    if (logs.ctrl) { try { logs.ctrl.abort(); } catch (e) {} logs.ctrl = null; }
    var ctrl = new AbortController();
    logs.ctrl = ctrl;
    logSetState('连接中');
    var res;
    try {
      res = await fetch(logUrl(), { cache: 'no-store', signal: ctrl.signal });
    } catch (e) {
      logs.ctrl = null;
      if (e && e.name === 'AbortError') { logSetState('已暂停'); return; }
      logFail('无法连接 /v1/logs/stream');
      return;
    }
    if (authRedirect(res)) { logs.ctrl = null; return; }
    if (!res.ok || !res.body) {
      logs.ctrl = null;
      logFail(res.status === 404 ? '这个 cwick 没有日志端点' : 'HTTP ' + res.status);
      return;
    }
    logs.fails = 0;
    logSetState('实时');

    var reader = res.body.getReader();
    var dec = new TextDecoder('utf-8');
    var buf = '';
    try {
      for (;;) {
        var chunk = await reader.read();
        if (chunk.done) break;
        buf += dec.decode(chunk.value, { stream: true });
        var idx;
        while ((idx = buf.indexOf('\n\n')) >= 0) {
          logFrame(buf.slice(0, idx));
          buf = buf.slice(idx + 2);
        }
      }
    } catch (e) {
      /* Server closed the connection (shutdown) or the socket dropped; in
         both cases falling through to the retry path is the right answer. */
    }
    logs.ctrl = null;
    if (ctrl.signal.aborted) { logSetState('已暂停'); return; }
    logSetState('已断开');
    logRetry();
  }

  function logFail(why) {
    logs.fails++;
    if (logs.fails >= 3) {
      logs.paused = true;
      syncLogPause();
      logSetState(why + ' · 已停止重连，点“继续”再试');
      return;
    }
    logSetState(why + ' · 重连中');
    logRetry();
  }

  function logRetry() {
    if (logs.paused || state.view !== 'logs') return;
    if (logs.timer) clearTimeout(logs.timer);
    var wait = Math.min(15000, 1200 * Math.pow(2, logs.fails));
    logs.timer = setTimeout(function () {
      logs.timer = 0;
      if (!logs.paused && state.view === 'logs') openLogStream();
    }, wait);
  }

  function syncLogPause() {
    var b = $('logPause');
    b.textContent = logs.paused ? '继续' : '暂停';
    b.classList.toggle('is-on', logs.paused);
  }

  function startLogs() {
    logFoot();
    if (!logs.paused && !logs.ctrl) openLogStream();
  }

  function stopLogs() {
    if (logs.timer) { clearTimeout(logs.timer); logs.timer = 0; }
    if (logs.ctrl) { try { logs.ctrl.abort(); } catch (e) {} logs.ctrl = null; }
    if (logs.raf) { cancelAnimationFrame(logs.raf); logs.raf = 0; }
    if (logs.pending.length) logFlush();
    logSetState(logs.paused ? '已暂停' : '已停止');
  }

  function togglePause() {
    logs.paused = !logs.paused;
    syncLogPause();
    if (logs.paused) stopLogs();
    else { logs.fails = 0; openLogStream(); }
  }

  function logClear() {
    logs.lines = [];
    logs.pending = [];
    logs.gap = 0;
    $('logBody').innerHTML =
      '<div class="log-empty">已清屏（只影响这个页面，服务端的环形缓冲仍在）。</div>';
    logFoot();
  }

  function logSave() {
    var txt = logs.lines.map(function (l) {
      return logTime(l.t) + ' ' + (l.src === 'err' ? 'ERR' : 'OUT') + ' ' + l.msg;
    }).join('\n');
    var a = document.createElement('a');
    a.href = URL.createObjectURL(
      new Blob([txt || '(empty)\n'], { type: 'text/plain;charset=utf-8' }));
    a.download = 'cwick-' + new Date().toISOString().replace(/[:.]/g, '-') + '.log';
    document.body.appendChild(a);
    a.click();
    setTimeout(function () { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
    setStatus('已导出 ' + logs.lines.length + ' 行日志');
  }

  function loadLogPrefs() {
    try {
      var p = JSON.parse(localStorage.getItem(LS_LOGPREFS) || 'null');
      if (!p) return;
      if (typeof p.follow === 'boolean') logs.follow = p.follow;
      if (typeof p.wrap === 'boolean') logs.wrap = p.wrap;
      if (p.src === 'all' || p.src === 'out' || p.src === 'err') logs.src = p.src;
    } catch (e) { /* private mode: defaults are fine */ }
  }
  function saveLogPrefs() {
    try {
      localStorage.setItem(LS_LOGPREFS, JSON.stringify({
        follow: logs.follow, wrap: logs.wrap, src: logs.src
      }));
    } catch (e) {}
  }
  function restoreLogControls() {
    $('logSrc').value = logs.src;
    $('logFollow').checked = logs.follow;
    $('logWrap').checked = logs.wrap;
    $('logBody').classList.toggle('wrap', logs.wrap);
    syncLogPause();
    logRenderAll();
    logFoot();
  }

  /* ------------------------------------------------------------------ */
  /* settings — read-only run config + switching the resident model       */
  /* ------------------------------------------------------------------ */
  /* Lifetime choices (port, slots, context window, byte caps) are set on the
     command line and rebuilding them at runtime would be a lie, so this page
     shows them and says so. What *can* change live — which local model is
     resident, and the per-request knobs this browser sends — is a control. */
  var catalog = [];

  function fmtTok(n) { return n ? fmtNum(n) + ' tok' : '—'; }

  function renderConfig(c) {
    c = c || {};
    $('cfgListen').textContent = (c.host || '?') + ':' + (c.port || '?');
    $('cfgSlots').textContent = c.slots == null ? '—' : c.slots;
    $('cfgCtx').textContent = fmtTok(c.context_window);
    $('cfgMaxTok').textContent = fmtTok(c.max_tokens);
    $('cfgReqBytes').textContent = c.max_request_bytes ? fmtBytes(c.max_request_bytes) : '—';
    $('cfgPfx').textContent = (c.prefix_cache && c.prefix_cache.enabled)
      ? '开 · block ' + (c.prefix_cache.block_tokens || 0) + ' tok' : '关';
    $('cfgLogs').textContent = c.log_capture
      ? '开 · 缓存 ' + (c.log_retained || 0) + ' 行' : '关';
    $('cfgFlags').textContent = [
      c.web ? 'web' : '—', c.verbose ? 'verbose' : '—', c.trace ? 'trace' : '—'
    ].join(' / ');
    $('cfgLibrary').textContent = c.model_library || '—';
    $('cfgCatalog').textContent = c.model_catalog || '—';
    var dirs = c.models_dirs || [];
    $('cfgDirs').innerHTML = dirs.length
      ? dirs.map(function (d) { return '<div>' + esc(d) + '</div>'; }).join('')
      : '<div class="dir-none">（未用 --models-dir 追加，只有启动时加载的那一个）</div>';
  }

  function renderModels(list) {
    catalog = list || [];
    var box = $('modelList');
    box.innerHTML = '';
    if (!catalog.length) {
      box.innerHTML = '<div class="model-row muted">没有找到本地模型。</div>';
      return;
    }
    catalog.forEach(function (m) {
      var cw = m.cwick || {};
      var loaded = !!cw.loaded;
      /* Older servers omit the field; treat those as switchable so the UI
       * stays usable against a binary that has not been rebuilt yet. */
      var supported = cw.supported !== false;
      var why = cw.unsupported_reason || '该模型架构没有对应的推理内核';
      var row = document.createElement('div');
      row.className = 'model-row' + (loaded ? ' is-loaded' : '');
      row.dataset.id = m.id;
      row.innerHTML =
        '<div class="mr-main">' +
          '<div class="mr-name">' + esc(m.id) + '</div>' +
          '<div class="mr-root" title="' + esc(supported ? '' : why) + '">' +
            esc(cw.root || '') + '</div>' +
        '</div><div class="mr-side">' +
          (loaded ? '<span class="tag-loaded">常驻</span>' : '') +
          (supported ? '' : '<span class="hub-badge warn" title="' +
            esc(why) + '">不支持</span>') +
          '<button class="btn-switch" type="button"' +
            ((loaded || !supported) ? ' disabled' : '') +
          '>' + (loaded ? '使用中' : (supported ? '切换' : '不支持')) +
          '</button></div>';
      box.appendChild(row);
    });
  }

  function setSwitchNote(text, cls) {
    var n = $('switchNote');
    n.hidden = !text;
    n.className = 'note' + (cls ? ' ' + cls : '');
    n.textContent = text || '';
  }

  async function loadModelList() {
    try {
      var r = await fetch('/v1/models', { cache: 'no-store' });
      if (authRedirect(r)) return;
      var j = await r.json();
      renderModels(j.data || []);
    } catch (e) {
      $('modelList').innerHTML =
        '<div class="model-row muted">无法读取 /v1/models</div>';
    }
  }

  function imageNote(text, cls) {
    var n = $('imageNote');
    if (!n) return;
    n.hidden = !text;
    n.className = 'note' + (cls ? ' ' + cls : '');
    n.textContent = text || '';
  }

  async function loadImageModelStatus() {
    if (!$('imageStatePill')) return;
    try {
      var r = await fetch('/v1/config', { cache: 'no-store' });
      if (authRedirect(r)) return;
      var j = await r.json();
      var im = j.image_model || {};
      var dir = im.dir || '';
      var loaded = !!im.loaded;
      $('imageDir').textContent = dir || '—';
      $('imageStatePill').textContent = t(loaded ? 'set.imageReady' : 'set.imageIdle');
      $('imageStatePill').classList.toggle('is-ready', loaded);
      var btn = $('imageSwitchBtn');
      if (btn) {
        btn.disabled = !dir;
        btn.textContent = t(loaded ? 'set.imageReady' : 'set.imageLoad');
      }
      if (!dir) imageNote(t('set.noImage'), '');
    } catch (e) { /* settings still usable without the image block */ }
  }

  async function switchImageModel() {
    var btn = $('imageSwitchBtn');
    imageNote(t('set.imageSwitching'), '');
    if (btn) btn.disabled = true;
    try {
      var res = await fetch('/v1/models/switch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: 'MiniMax-H3-Image' })
      });
      if (authRedirect(res)) return;
      var j = {};
      try { j = await res.json(); } catch (e) {}
      var detail = (j.error && j.error.message) || j.detail || ('HTTP ' + res.status);
      if (res.ok) imageNote(t('set.imageOk') + ' (' + detail + ')', 'ok');
      else imageNote(t('set.imageFail') + detail, 'bad');
      setStatus(res.ok ? 'MiniMax-H3-Image' : 'HTTP ' + res.status);
    } catch (e) {
      imageNote(t('set.imageFail') + (e && e.message ? e.message : e), 'bad');
    }
    await loadImageModelStatus();
  }

  async function loadSettings() {
    try {
      var rc = await fetch('/v1/config', { cache: 'no-store' });
      if (authRedirect(rc)) return;
      renderConfig(await rc.json());
    } catch (e) {
      setStatus('无法读取 /v1/config');
    }
    await loadModelList();
    await loadImageModelStatus();
    await loadAuthState();
    await loadKeys();
  }

  async function switchModel(id) {
    setSwitchNote('正在校验并切换到 ' + id + '…', '');
    var btns = document.querySelectorAll('#modelList .btn-switch');
    for (var i = 0; i < btns.length; i++) btns[i].disabled = true;
    try {
      var res = await fetch('/v1/models/switch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: id })
      });
      if (authRedirect(res)) return;
      var j = {};
      try { j = await res.json(); } catch (e) {}
      var detail = (j.error && j.error.message) || j.detail || ('HTTP ' + res.status);
      if (res.ok) {
        setSwitchNote('✓ ' + detail, 'ok');
        setStatus('模型已切换：' + id);
      } else {
        setSwitchNote('✗ ' + detail +
          (res.status === 409 ? ' — 等当前请求结束再试一次。' : ''), 'bad');
        setStatus('切换失败：' + id);
      }
    } catch (e) {
      setSwitchNote('✗ 无法连接服务端：' + (e && e.message ? e.message : e), 'bad');
    }
    /* The engine, the tokenizer and the metrics identity all changed. */
    await loadSettings();
    await loadModel();
    loadMetrics(true);
  }

  /* ------------------------------------------------------------------ */
  /* model library — local inventory, live downloads, remote catalog    */
  /* ------------------------------------------------------------------ */
  var hub = { timer: 0, tick: 0, local: [], jobs: [], hits: [], root: '', endpoint: '' };

  function hubLocalNote(text, bad) {
    var n = $('hubLocalNote');
    n.hidden = !text;
    n.className = 'note' + (bad ? ' bad' : '');
    n.textContent = text || '';
  }
  function hubSearchNote(text, bad) {
    var n = $('hubSearchNote');
    n.hidden = !text;
    n.className = 'note' + (bad ? ' bad' : '');
    n.textContent = text || '';
  }
  async function hubReq(url, opts) {
    var res = await fetch(url, Object.assign({ cache: 'no-store' }, opts || {}));
    if (authRedirect(res)) return null;
    var j = {};
    try { j = await res.json(); } catch (e) {}
    if (!res.ok) throw new Error((j.error && j.error.message) || ('HTTP ' + res.status));
    return j;
  }
  function hubStateBadge(s) {
    var map = {
      queued: ['排队中', 'info'], listing: ['读取文件列表', 'info'],
      running: ['下载中', 'info'], paused: ['已暂停', 'warn'],
      done: ['已完成', 'ok'], error: ['失败', 'bad'], canceled: ['已取消', 'warn']
    };
    var e = map[s] || [s, ''];
    return '<span class="hub-badge ' + e[1] + '">' + esc(e[0]) + '</span>';
  }
  function hubEta(sec) {
    if (sec == null || sec < 0 || !isFinite(sec)) return '—';
    if (sec < 60) return Math.round(sec) + ' 秒';
    if (sec < 3600) return Math.round(sec / 60) + ' 分钟';
    if (sec < 86400) return (sec / 3600).toFixed(1) + ' 小时';
    return (sec / 86400).toFixed(1) + ' 天';
  }

  function hubRenderLocal() {
    var box = $('hubLocal');
    $('hubRootLabel').textContent = hub.root || './models';
    $('hubEndpointLabel').textContent = hub.endpoint || 'hf-mirror';
    if (!hub.local.length) {
      box.innerHTML = '<div class="model-row muted">这个目录下还没有模型。用下面的检索下载一个。</div>';
      return;
    }
    box.innerHTML = hub.local.map(function (m) {
      var ready = !!m.complete;
      var huge = m.bytes >= 100 * 1073741824;
      var supported = m.supported !== false;
      var image = m.kind === 'image';
      var video = m.kind === 'video';
      var served = m.served_id || m.id;
      var why = m.unsupported_reason || t('hub.archMismatch');
      return '<div class="hub-card ' + (ready ? 'is-ready' : 'is-partial') + '">' +
        '<div class="hub-title">' + esc(m.id) +
          (ready ? '<span class="hub-badge ok">' + t('hub.ready') + '</span>'
                 : '<span class="hub-badge warn">' + t('hub.partial') + '</span>') +
          (image ? '<span class="hub-badge info" title="' + esc(t('set.imageTip')) +
            '">' + t('hub.kindImage') + '</span>' : '') +
          (video ? '<span class="hub-badge info" title="' + esc(t('set.videoTip')) +
            '">' + t('hub.kindVideo') + '</span>' : '') +
          (ready && !supported
            ? '<span class="hub-badge warn" title="' + esc(why) + '">' +
              t('hub.unsupported') + '</span>' : '') +
          (huge ? '<span class="hub-badge info">' + t('hub.huge') + '</span>' : '') +
        '</div>' +
        '<div class="hub-sub">' + esc(m.repo) + ' · ' + esc(m.path) + '</div>' +
        '<div class="hub-meta">' +
          '<span>大小 <b>' + fmtBytes(m.bytes) + '</b></span>' +
          '<span>文件 <b>' + (m.files || 0) + '</b></span>' +
          (m.partial_files
            ? '<span>待续传 <b>' + m.partial_files + '</b> 个 / <b>' +
              fmtBytes(m.partial_bytes) + '</b></span>' : '') +
        '</div>' +
        '<div class="hub-actions">' +
          (ready && supported
            ? '<button class="min-btn accent" type="button" data-hub-switch="' +
              esc(served) + '"' + (image ? ' title="' + esc(t('set.imageTip')) + '"' :
                video ? ' title="' + esc(t('set.videoTip')) + '"' : '') +
              '>' + t(image ? 'hub.switchImage' : video ? 'hub.switchVideo' :
                'hub.switchResident') + '</button>'
            : (ready ? '<span class="hub-badge info" title="' + esc(why) +
                '">' + t('hub.archMismatch') + '</span>' : '')) +
          '<button class="min-btn danger" type="button" data-hub-delete="' +
            esc(m.repo) + '" title="删除本地权重（不可恢复）">删除</button>' +
        '</div></div>';
    }).join('');
  }

  function hubRenderJobs() {
    var box = $('hubJobs');
    $('hubPoll').textContent = hub.jobs.length ? hub.jobs.length + ' 个任务 · 1.5 s 刷新' : '';
    if (!hub.jobs.length) {
      box.innerHTML = '<div class="model-row muted">没有下载任务</div>';
      return;
    }
    box.innerHTML = hub.jobs.map(function (j) {
      var pct = Math.max(0, Math.min(100, (j.progress || 0) * 100));
      var waiting = (j.retry_in_seconds || 0) > 0;
      var cls = j.state === 'error' ? 'is-error' :
                (j.state === 'running' || j.state === 'listing' || j.state === 'queued')
                  ? 'is-running' : '';
      var bar = j.state === 'done' ? 'done' : (j.state === 'paused' ? 'paused' : '');
      var huge = j.bytes_total >= 100 * 1073741824;
      var acts = '';
      if (j.state === 'running' || j.state === 'listing' || j.state === 'queued') {
        acts += '<button class="min-btn" type="button" data-hub-act="pause" data-hub-id="' +
                esc(j.id) + '">暂停</button>';
      } else if (j.state === 'paused') {
        acts += '<button class="min-btn accent" type="button" data-hub-act="resume" data-hub-id="' +
                esc(j.id) + '">继续下载</button>';
      } else if (j.state === 'error') {
        acts += '<button class="min-btn accent" type="button" data-hub-act="retry" data-hub-id="' +
                esc(j.id) + '">断点续传</button>';
      } else if (j.state === 'canceled') {
        acts += '<button class="min-btn accent" type="button" data-hub-act="resume" data-hub-id="' +
                esc(j.id) + '">继续下载</button>';
      } else {
        acts += '<button class="min-btn" type="button" data-hub-act="resume" data-hub-id="' +
                esc(j.id) + '">重新校验</button>';
      }
      if (j.state !== 'done' && j.state !== 'canceled') {
        acts += '<button class="min-btn danger" type="button" data-hub-act="cancel" data-hub-id="' +
                esc(j.id) + '">取消</button>';
      }
      if (j.state === 'done') {
        acts += '<button class="min-btn accent" type="button" data-hub-switch="' +
                esc(j.repo) + '">设为常驻</button>';
      }
      /* A retry in flight is still a "running" job: say so explicitly, and
       * show the countdown, so a backoff pause is not mistaken for a hang. */
      var retryBadge = waiting
        ? '<span class="hub-badge warn">自动重试 ' + (j.attempts || 0) + '/' +
          (j.max_attempts || 0) + ' · ' + j.retry_in_seconds + ' s 后</span>'
        : ((j.attempts || 0) > 0 && j.state !== 'done'
            ? '<span class="hub-badge info">已重试 ' + j.attempts + ' 次</span>' : '');
      return '<div class="hub-job ' + cls + '">' +
        '<div class="hub-title">' + esc(j.repo) + hubStateBadge(j.state) +
          (huge ? '<span class="hub-badge info">超大权重</span>' : '') +
          (j.range_resume ? '<span class="hub-badge warn">已保留断点</span>' : '') +
          retryBadge +
        '</div>' +
        '<div class="hub-meta">' +
          '<span>进度 <b>' + pct.toFixed(1) + '%</b></span>' +
          '<span>已下载 <b>' + fmtBytes(j.bytes_done) + '</b> / ' + fmtBytes(j.bytes_total) + '</span>' +
          '<span>分片 <b>' + (j.files_done || 0) + '/' + (j.files_total || 0) + '</b></span>' +
          '<span>速率 <b>' + (j.speed_bps ? fmtBytes(j.speed_bps) + '/s' : '—') + '</b></span>' +
          '<span>剩余 <b>' + hubEta(j.eta_seconds) + '</b></span>' +
        '</div>' +
        '<div class="hub-bar"><i class="' + bar + '" style="width:' + pct.toFixed(2) + '%"></i></div>' +
        (j.current ? '<div class="hub-cur">当前文件 ' + esc(j.current) + '</div>' : '') +
        (j.error ? '<div class="hub-error">' + esc(j.error) + '</div>' : '') +
        '<div class="hub-actions">' + acts + '</div>' +
      '</div>';
    }).join('');
  }

  function hubRenderHits() {
    var box = $('hubResults');
    if (!hub.hits.length) {
      box.innerHTML = '<div class="model-row muted">没有匹配的模型</div>';
      return;
    }
    box.innerHTML = hub.hits.map(function (m) {
      var tags = (m.tags || []).slice(0, 3).map(function (t) {
        return '<span>' + esc(t) + '</span>';
      }).join('');
      return '<div class="hub-hit">' +
        '<div class="hh-main">' +
          '<div class="hh-id">' + esc(m.id) +
            (m.gated ? ' <span class="hub-badge warn">需授权</span>' : '') +
            (m.private ? ' <span class="hub-badge bad">私有</span>' : '') +
          '</div>' +
          '<div class="hh-meta"><span>↓ ' + fmtNum(m.downloads) + '</span>' +
            '<span>♥ ' + fmtNum(m.likes) + '</span>' +
            (m.pipeline_tag ? '<span>' + esc(m.pipeline_tag) + '</span>' : '') +
            tags + '</div>' +
        '</div>' +
        '<button class="min-btn accent" type="button" data-hub-start="' + esc(m.id) + '">下载</button>' +
      '</div>';
    }).join('');
  }

  async function hubLoadLocal() {
    try {
      var j = await hubReq('/v1/hub/models');
      if (!j) return;
      hub.local = j.models || [];
      hub.root = j.root || hub.root;
      hub.endpoint = j.endpoint || hub.endpoint;
      hubRenderLocal();
    } catch (e) {
      hubLocalNote('读取本地模型失败：' + (e && e.message ? e.message : e), true);
    }
  }
  async function hubLoadJobs() {
    try {
      var j = await hubReq('/v1/hub/downloads');
      if (!j) return;
      hub.jobs = j.jobs || [];
      hubRenderJobs();
    } catch (e) {
      $('hubJobs').innerHTML = '<div class="model-row muted">无法读取下载队列</div>';
    }
  }
  function hubRefresh() { hubLoadLocal(); hubLoadJobs(); }

  function hubStopPolling() {
    if (hub.timer) { clearInterval(hub.timer); hub.timer = 0; }
    var p = $('hubPoll');
    if (p) p.textContent = '';
  }
  function hubStartPolling() {
    hubStopPolling();
    hubRefresh();
    hub.timer = setInterval(function () {
      if (state.view !== 'models') return;
      hubLoadJobs();
      if (++hub.tick % 4 === 0) hubLoadLocal();
    }, 1500);
  }

  async function hubSearch() {
    var q = ($('hubQuery').value || '').trim();
    if (!q) { hubSearchNote('输入关键词再检索。', false); return; }
    var btn = $('hubSearchBtn');
    btn.disabled = true;
    hubSearchNote('正在检索…', false);
    try {
      var j = await hubReq('/v1/hub/search?q=' + encodeURIComponent(q) + '&limit=16');
      if (!j) return;
      hub.hits = j.models || [];
      hubRenderHits();
      hubSearchNote(j.error
        ? '检索失败：' + j.error
        : '找到 ' + hub.hits.length + ' 个模型 · 来源 ' + (j.endpoint || ''),
        !!j.error);
    } catch (e) {
      hubSearchNote('检索失败：' + (e && e.message ? e.message : e), true);
    } finally {
      btn.disabled = false;
    }
  }

  async function hubStart(repo) {
    try {
      hubLocalNote('正在启动下载：' + repo, false);
      var j = await hubReq('/v1/hub/download', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ repo: repo })
      });
      if (!j) return;
      hubLocalNote('已加入下载队列：' + repo + '（' + j.id + '）', false);
      await hubLoadJobs();
    } catch (e) {
      hubLocalNote('无法开始下载：' + (e && e.message ? e.message : e), true);
    }
  }

  async function hubAction(id, act) {
    try {
      var j = await hubReq('/v1/hub/download/' + encodeURIComponent(id) + '/' + act,
                           { method: 'POST' });
      if (!j) return;
      hubLocalNote('任务 ' + id + '：' + act, false);
      await hubLoadJobs();
    } catch (e) {
      hubLocalNote('操作失败：' + (e && e.message ? e.message : e), true);
    }
  }

  async function hubDelete(repo) {
    var m = (hub.local || []).filter(function (x) {
      return x.repo === repo || x.id === repo;
    })[0];
    var files = m ? (m.files || 0) + (m.partial_files || 0) : 0;
    var bytes = m ? (m.bytes || 0) + (m.partial_bytes || 0) : 0;
    var what = (files ? files + ' 个文件' : '全部文件') +
               (bytes ? '、' + fmtBytes(bytes) : '');
    /* Irreversible, and the server deletes recursively: ask first, and say
     * exactly what is about to disappear. */
    if (!window.confirm('删除本地模型 ' + repo + '？\n\n将移除 ' + what +
                        '，此操作不可恢复。')) return;
    hubLocalNote('正在删除 ' + repo + '…', false);
    try {
      var j = await hubReq('/v1/hub/models/' + encodeURIComponent(repo),
                           { method: 'DELETE' });
      if (!j) return;
      hubLocalNote('✓ 已删除 ' + repo + '（释放 ' + fmtBytes(j.bytes || 0) + '）', false);
      setStatus('已删除本地模型：' + repo);
      await hubLoadLocal();
      await loadModelList();
      await loadModel();
    } catch (e) {
      hubLocalNote('✗ 删除失败：' + (e && e.message ? e.message : e), true);
    }
  }

  async function hubSwitch(repo) {
    /* The completed-download list has a switch button too, but only the
     * inventory carries the compatibility verdict. Check here as well so a
     * click there never sends a request the server must refuse. */
    var local = (hub.local || []).filter(function (x) {
      return x.repo === repo || x.id === repo || x.served_id === repo;
    })[0];
    if (local && local.supported === false) {
      hubLocalNote('✗ 未发起切换：' +
        (local.unsupported_reason || '该模型架构没有对应的推理内核'), true);
      return;
    }
    var target = (local && local.served_id) ? local.served_id : repo;
    hubLocalNote('正在校验并切换到 ' + repo + '…', false);
    try {
      var j = await hubReq('/v1/models/switch', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: target })
      });
      if (!j) return;
      hubLocalNote('✓ 已切换到 ' + repo, false);
      setStatus('模型已切换：' + repo);
      await loadModel();
      await hubLoadLocal();
      if (local && local.kind === 'image') await loadImageModelStatus();
    } catch (e) {
      hubLocalNote('✗ 切换失败：' + (e && e.message ? e.message : e), true);
    }
  }

  /* ------------------------------------------------------------------ */
  /* view / wiring                                                       */
  /* ------------------------------------------------------------------ */
  var VIEW_META = {
    chat: { title: 'view.chat.title', sub: 'view.chat.sub' },
    apps: { title: 'view.apps.title', sub: 'view.apps.sub' },
    models: { title: 'view.models.title', sub: 'view.models.sub' },
    logs: { title: 'view.logs.title', sub: 'view.logs.sub' },
    settings: { title: 'view.settings.title', sub: 'view.settings.sub' }
  };
  function showView(v) {
    if (!VIEW_META[v]) v = 'chat';
    state.view = v;
    $('chatView').hidden = v !== 'chat';
    $('appsView').hidden = v !== 'apps';
    $('modelsView').hidden = v !== 'models';
    $('logsView').hidden = v !== 'logs';
    $('settingsView').hidden = v !== 'settings';
    $('sessionSection').style.display = v === 'chat' ? '' : 'none';
    $('viewTitle').textContent = t(VIEW_META[v].title);
    $('viewSub').textContent = t(VIEW_META[v].sub);
    var items = document.querySelectorAll('.nav-item');
    for (var i = 0; i < items.length; i++) {
      items[i].classList.toggle('is-active', items[i].dataset.view === v);
    }

    /* Each view owns exactly one live feed, and none of them should keep
       pulling while the user is looking at another tab. */
    if (v === 'apps') { loadApps(false); appsSyncView(); }
    if (v === 'models') hubStartPolling(); else hubStopPolling();
    if (v === 'logs') startLogs(); else stopLogs();
    if (v === 'settings') {
      loadSettings(); loadMetrics(true); trendDrawAll(); pmDrawAll();
    }

    /* Deep-linkable: #logs / #settings open straight onto that tab. */
    try {
      history.replaceState(null, '', v === 'chat' ? location.pathname : '#' + v);
    } catch (e) { /* file:// or a locked-down browser: not important */ }
  }

  function autoGrow() {
    var ta = $('input');
    ta.style.height = 'auto';
    ta.style.height = Math.min(220, ta.scrollHeight) + 'px';
  }

  function submitComposer() {
    if (state.busy || state.sending || state.imageBusy) return;
    var text = $('input').value;
    var images = state.draftImages.slice();
    if (!text.trim() && !images.length) return;
    $('input').value = '';
    state.draftImages = [];
    autoGrow();
    renderDraftImages();
    send(text, images);
  }

  function bind() {
    $('sendBtn').addEventListener('click', submitComposer);
    $('stopBtn').addEventListener('click', stop);
    $('leaveQueue').addEventListener('click', stop);

    $('input').addEventListener('keydown', function (e) {
      if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
        e.preventDefault();
        submitComposer();
      }
    });
    $('input').addEventListener('input', autoGrow);
    $('input').addEventListener('paste', function (e) {
      var files = Array.prototype.slice.call((e.clipboardData && e.clipboardData.items) || [])
        .filter(function (item) { return item.kind === 'file'; })
        .map(function (item) { return item.getAsFile(); }).filter(Boolean);
      if (files.length) { e.preventDefault(); addImageFiles(files); }
    });

    $('attachBtn').addEventListener('click', function () { $('imageInput').click(); });
    $('imageInput').addEventListener('change', function () {
      addImageFiles(this.files);
      this.value = '';
    });
    $('draftImages').addEventListener('click', function (e) {
      var button = e.target.closest('[data-remove-image]');
      if (button && !button.disabled) removeDraftImage(button.getAttribute('data-remove-image'));
    });
    $('composer').addEventListener('dragover', function (e) {
      if (e.dataTransfer && Array.prototype.indexOf.call(e.dataTransfer.types || [], 'Files') >= 0)
        e.preventDefault();
    });
    $('composer').addEventListener('drop', function (e) {
      if (!e.dataTransfer || !e.dataTransfer.files || !e.dataTransfer.files.length) return;
      e.preventDefault();
      addImageFiles(e.dataTransfer.files);
    });

    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape') stop();
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        newSession();
      }
    });

    $('newChat').addEventListener('click', function () { newSession(); });

    $('sessionList').addEventListener('click', function (e) {
      var del = e.target.closest('.del');
      var row = e.target.closest('.session');
      if (!row || !row.dataset.id) return;
      if (del) {
        if (state.busy && row.dataset.id === state.currentId) {
          setStatus(t('chat.busyDelete'));
          return;
        }
        var removed = state.sessions.filter(function (s) { return s.id === row.dataset.id; });
        state.sessions = state.sessions.filter(function (s) { return s.id !== row.dataset.id; });
        if (state.currentId === row.dataset.id) state.currentId = state.sessions[0] ? state.sessions[0].id : null;
        discardImages(removed.reduce(function (all, s) { return all.concat(sessionImages(s)); }, []));
        saveSessions();
        renderSessions();
        renderMessages();
        return;
      }
      state.currentId = row.dataset.id;
      renderSessions();
      renderMessages();
    });

    $('messages').addEventListener('click', function (e) {
      var imageBtn = e.target.closest('[data-remove-history-image]');
      if (imageBtn) {
        removeHistoryImage(imageBtn.getAttribute('data-remove-history-image'));
        return;
      }
      var act = e.target.closest('.code-block button[data-act]');
      if (act) {
        var block = act.closest('.code-block');
        var what = act.dataset.act;
        if (what === 'copy') copyText(blockText(block), act);
        else if (what === 'dl') downloadText(blockText(block), 'cwick-snippet.' + extFor(block));
        else if (what === 'run') loadRun(blockText(block));
        return;
      }
      var ex = e.target.closest('.example');
      if (ex) {
        $('input').value = ex.dataset.prompt || '';
        autoGrow();
        $('input').focus();
      }
    });

    var navs = document.querySelectorAll('.nav-item');
    for (var i = 0; i < navs.length; i++) {
      navs[i].addEventListener('click', function () { showView(this.dataset.view); });
    }

    /* Language switch: instant, no navigation, persisted in prefs. */
    var langs = document.querySelectorAll('#langSwitch .lang-btn');
    for (var li = 0; li < langs.length; li++) {
      langs[li].addEventListener('click', function () { setLang(this.getAttribute('data-lang')); });
    }
    var imgBtn = $('imageSwitchBtn');
    if (imgBtn) imgBtn.addEventListener('click', switchImageModel);

    $('apiSel').addEventListener('change', function () {
      state.prefs.api = normalizeApi(this.value);
      savePrefs();
      $('setApi').value = state.prefs.api;
      updateApiMode();
    });
    $('effort').addEventListener('change', function () {
      state.prefs.effort = this.value; savePrefs();
      $('setEffort').value = this.value;
    });
    $('maxTokens').addEventListener('change', function () {
      state.prefs.maxTokens = Number(this.value) || 2048; savePrefs();
      $('setMaxTokens').value = state.prefs.maxTokens;
    });
    $('streamToggle').addEventListener('change', function () {
      state.prefs.stream = this.checked; savePrefs();
      $('setStream').checked = this.checked;
    });
    $('toolToggle').addEventListener('change', function () {
      state.prefs.tools = this.checked; savePrefs();
      $('setTools').checked = this.checked;
    });
    $('liveToggle').addEventListener('change', function () {
      startLive(this.checked);
    });

    /* ---- logs ---- */
    $('logSrc').addEventListener('change', function () {
      logs.src = this.value; saveLogPrefs(); logRenderAll();
    });
    $('logQuery').addEventListener('input', function () {
      logs.query = this.value.trim().toLowerCase(); logRenderAll();
    });
    $('logWrap').addEventListener('change', function () {
      logs.wrap = this.checked; saveLogPrefs();
      $('logBody').classList.toggle('wrap', logs.wrap);
    });
    $('logFollow').addEventListener('change', function () {
      logs.follow = this.checked; saveLogPrefs();
      if (logs.follow) $('logBody').scrollTop = $('logBody').scrollHeight;
    });
    $('logPause').addEventListener('click', togglePause);
    $('logClear').addEventListener('click', logClear);
    $('logSave').addEventListener('click', logSave);

    /* ---- model library ---- */
    $('hubRefresh').addEventListener('click', function () { hubRefresh(); });
    $('hubSearchBtn').addEventListener('click', hubSearch);
    $('hubQuery').addEventListener('keydown', function (e) {
      if (e.key === 'Enter') { e.preventDefault(); hubSearch(); }
    });
    $('hubLocal').addEventListener('click', function (e) {
      var b = e.target.closest('[data-hub-switch]');
      if (b && !b.disabled) { hubSwitch(b.dataset.hubSwitch); return; }
      var d = e.target.closest('[data-hub-delete]');
      if (d && !d.disabled) hubDelete(d.dataset.hubDelete);
    });
    $('hubJobs').addEventListener('click', function (e) {
      var sw = e.target.closest('[data-hub-switch]');
      if (sw && !sw.disabled) { hubSwitch(sw.dataset.hubSwitch); return; }
      var b = e.target.closest('[data-hub-act]');
      if (b && !b.disabled) hubAction(b.dataset.hubId, b.dataset.hubAct);
    });
    $('hubResults').addEventListener('click', function (e) {
      var b = e.target.closest('[data-hub-start]');
      if (b && !b.disabled) hubStart(b.dataset.hubStart);
    });

    /* ---- settings ---- */
    $('modelsRefresh').addEventListener('click', function () { loadSettings(); });
    $('configRefresh').addEventListener('click', function () { loadSettings(); });
    $('modelList').addEventListener('click', function (e) {
      var btn = e.target.closest('.btn-switch');
      if (!btn || btn.disabled) return;
      var row = e.target.closest('.model-row');
      if (row && row.dataset.id) switchModel(row.dataset.id);
    });
    $('setApi').addEventListener('change', function () {
      state.prefs.api = normalizeApi(this.value);
      savePrefs();
      $('apiSel').value = state.prefs.api;
      updateApiMode();
    });
    $('setEffort').addEventListener('change', function () {
      state.prefs.effort = this.value; savePrefs();
      $('effort').value = this.value;
    });
    $('setMaxTokens').addEventListener('change', function () {
      state.prefs.maxTokens = Number(this.value) || 2048; savePrefs();
      $('maxTokens').value = state.prefs.maxTokens;
    });
    $('setStream').addEventListener('change', function () {
      state.prefs.stream = this.checked; savePrefs();
      $('streamToggle').checked = this.checked;
    });
    $('setTools').addEventListener('change', function () {
      state.prefs.tools = this.checked; savePrefs();
      $('toolToggle').checked = this.checked;
    });
    $('apiHelp').addEventListener('click', function (e) {
      e.preventDefault();
      $('apiDialog').showModal();
    });

    /* The trend plot is in-memory: clearing it only drops the drawn history,
       it never touches the server's metrics or the model. */
    $('trendClear').addEventListener('click', trendClear);
    $('pmQuery').addEventListener('input', function () {
      pm.query = this.value.trim().toLowerCase();
      pmRenderList();
    });
    $('pmRate').addEventListener('change', function () {
      pm.rate = this.checked; pmSave(); pmResetHist();
    });
    $('pmClear').addEventListener('click', function () {
      pm.selected = []; pmSave(); pmRenderList(); pmRenderCharts();
    });
    $('pmList').addEventListener('click', function (e) {
      var node = e.target.closest('[data-pm]');
      if (node && node.dataset.pm) pmToggle(node.dataset.pm);
    });
    $('liveToggle').addEventListener('change', function () {
      if (this.checked) loadMetrics(true);
    });
    window.addEventListener('resize', function () { trendDrawAll(); pmDrawAll(); });
    if ($('logoutBtn')) $('logoutBtn').addEventListener('click', logout);
    if ($('keyCreate')) $('keyCreate').addEventListener('click', createKey);
    if ($('keyRefresh')) $('keyRefresh').addEventListener('click', loadKeys);
    if ($('keyCopy')) $('keyCopy').addEventListener('click', copyKeySecret);
    if ($('keyList')) $('keyList').addEventListener('click', function (e) {
      var b = e.target.closest('[data-revoke]');
      if (b) revokeKey(b.dataset.revoke);
    });
  }

  function startLive(on) {
    if (state.liveTimer) { clearInterval(state.liveTimer); state.liveTimer = 0; }
    if (on) state.liveTimer = setInterval(function () { loadMetrics(false); }, 2000);
  }

  function restoreControls() {
    $('apiSel').value = apiKind();
    $('effort').value = state.prefs.effort;
    $('maxTokens').value = state.prefs.maxTokens;
    $('streamToggle').checked = !!state.prefs.stream;
    $('toolToggle').checked = !!state.prefs.tools;
    $('setApi').value = apiKind();
    $('setEffort').value = state.prefs.effort;
    $('setMaxTokens').value = state.prefs.maxTokens;
    $('setStream').checked = !!state.prefs.stream;
    updateApiMode();
    $('setTools').checked = !!state.prefs.tools;
    $('pmRate').checked = !!pm.rate;
    pmSyncSelNote();
    pmRenderCharts();
  }

  function boot() {
    /* Keep the served empty state as a string: renderMessages() wipes the
       container, and a cloned node would lose the delegated listeners it
       never had anyway. */
    loadPrefs();
    applyI18n();
    /* Capture after translation so a later renderMessages() cannot restore
     * the served (Chinese) copy over an English empty state. */
    EMPTY_HTML = $('emptyState') ? $('emptyState').outerHTML : '';
    pmLoad();
    loadLogPrefs();
    loadSessions();
    pruneOrphanImages();
    restoreControls();
    restoreLogControls();
    bind();
    appsBind();

    if (!state.sessions.length) newSession();
    else state.currentId = state.sessions[0].id;

    renderSessions();
    renderMessages();
    showView((location.hash || '').replace(/^#/, ''));
    loadModel();
    loadMetrics(true);
    startLive(true);
    /* the rail's memory bar must stay live even while settings is hidden */
    state.railTimer = setInterval(function () {
      if (!state.liveTimer) loadMetrics(false);
    }, 5000);
    autoGrow();
    $('input').focus();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
