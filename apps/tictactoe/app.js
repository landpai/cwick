/* cwick app: 3x3 tic-tac-toe against this server's decision endpoint.
 *
 * The app is self-contained: it is served from /apps/tictactoe/ by the cwick
 * binary, and sends one /v1/systemone decision per machine move: one prefill
 * of the board plus per-cell tactical facts, and the empty cell with the
 * highest probability of winning is the move. Nothing is generated, so there
 * is no JSON to parse and no prose to repair. The app counts board facts; it
 * never ranks cells -- the checkpoint's distribution over the option letters
 * is the decision, and its argmax is the move.
 * Every request is same-origin: the resident model is discovered from
 * /v1/models and re-discovered after a switch, so the endpoint's decision
 * adapter always follows the checkpoint that is actually loaded.
 *
 * Where the data lives:
 *   - the finished games, the move-by-move replay log and the scoreboard go
 *     to the server's sqlite file through /v1/apps/tictactoe/{state,events,games},
 *     which is why "clear site data" does not erase a match history the model
 *     produced and why the same history shows up on the phone that scanned the
 *     QR;
 *   - localStorage is written as well, and is the only storage when the server
 *     has no store (or when an app copy is opened from a file:// URL). Every
 *     write therefore has a fallback and every read prefers the server.
 */
(function () {
  'use strict';

  var $ = function (id) { return document.getElementById(id); };
  var LS_GAME = 'cwick.ttt.v1';
  var LINES = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8],
    [0, 3, 6], [1, 4, 7], [2, 5, 8],
    [0, 4, 8], [2, 4, 6]
  ];
  var S = {
    board: ['', '', '', '', '', '', '', '', ''],
    turn: 'X',
    over: false,
    winner: null,
    line: null,
    busy: false,
    moves: [],
    log: [],
    gen: 0,
    startedAt: 0,
    gameId: '',
    abort: null,
    timer: 0,
    modelId: '',
    adapter: null,
    score: { wins: 0, losses: 0, draws: 0, games: 0 },
    games: [],
    /* Server-side history (newest first) and the one being replayed. */
    history: [],
    replay: null,
    sideTab: 'now',
    seq: 0
  };

  /* ---------------------------------------------------------------- */
  /* storage: server sqlite first, localStorage as the fallback        */
  /* ---------------------------------------------------------------- */
  var STORE = { app: 'tictactoe', ok: null };   /* null = not probed yet */

  function lsGet(key) {
    try {
      var j = JSON.parse(localStorage.getItem(LS_GAME) || '{}') || {};
      return j[key] != null ? j[key] : null;
    } catch (e) { return null; }
  }
  function lsPut(key, value) {
    try {
      var j = JSON.parse(localStorage.getItem(LS_GAME) || '{}') || {};
      j[key] = value;
      localStorage.setItem(LS_GAME, JSON.stringify(j));
      return true;
    } catch (e) { return false; }
  }

  /* Store calls are same-origin too: addressed by path, not by a
   * configurable base URL. */
  async function storeFetch(path, opts) {
    var res = await fetch(path, Object.assign({ cache: 'no-store' }, opts || {}));
    if (authRedirect(res)) throw new Error('authentication required');
    return res;
  }

  async function storeProbe() {
    try {
      var res = await storeFetch('/v1/store?app=tictactoe');
      var ok = res.ok;
      if (ok) { await res.json(); STORE.ok = true; }
      else STORE.ok = false;
    } catch (e) { STORE.ok = false; }
    storeBadge();
    return STORE.ok;
  }
  function storeBadge() {
    var el = $('storeBadge');
    if (!el) return;
    if (STORE.ok === true) { el.textContent = '存储 sqlite（服务端）'; el.className = 'chip mono ok'; }
    else if (STORE.ok === false) { el.textContent = '存储 localStorage（回退）'; el.className = 'chip mono warn'; }
    else { el.textContent = '存储 —'; el.className = 'chip mono'; }
  }

  async function storeGet(key) {
    if (STORE.ok === false) return lsGet(key);
    try {
      var res = await storeFetch('/v1/apps/' + STORE.app + '/state?key=' +
        encodeURIComponent(key));
      if (!res.ok) { STORE.ok = false; storeBadge(); return lsGet(key); }
      var j = await res.json();
      STORE.ok = true;
      return j && j.value != null ? j.value : null;
    } catch (e) { STORE.ok = false; storeBadge(); return lsGet(key); }
  }

  async function storePut(key, value) {
    lsPut(key, value);                    /* always mirror: offline must work */
    if (STORE.ok === false) return false;
    try {
      var res = await storeFetch('/v1/apps/' + STORE.app + '/state', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ key: key, value: value })
      });
      if (!res.ok) { STORE.ok = false; storeBadge(); return false; }
      STORE.ok = true;
      return true;
    } catch (e) { STORE.ok = false; storeBadge(); return false; }
  }

  async function storeEventAdd(gameId, seq, kind, payload) {
    if (STORE.ok === false) return false;
    try {
      var res = await storeFetch('/v1/apps/' + STORE.app + '/events', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ game_id: gameId, seq: seq, kind: kind, payload: payload })
      });
      if (!res.ok) { STORE.ok = false; storeBadge(); return false; }
      return true;
    } catch (e) { return false; }
  }

  async function storeEvents(gameId) {
    if (STORE.ok === false) return null;
    try {
      var res = await storeFetch('/v1/apps/' + STORE.app + '/events?game_id=' +
        encodeURIComponent(gameId) + '&limit=200');
      if (!res.ok) return null;
      var j = await res.json();
      return (j && Array.isArray(j.events)) ? j.events : null;
    } catch (e) { return null; }
  }

  async function storeGamePut(game) {
    if (STORE.ok === false) return false;
    try {
      var res = await storeFetch('/v1/apps/' + STORE.app + '/games', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(game)
      });
      if (!res.ok) { STORE.ok = false; storeBadge(); return false; }
      return true;
    } catch (e) { return false; }
  }

  async function storeGames() {
    if (STORE.ok === false) return null;
    try {
      var res = await storeFetch('/v1/apps/' + STORE.app + '/games?limit=20');
      if (!res.ok) return null;
      var j = await res.json();
      return (j && Array.isArray(j.games)) ? j.games : null;
    } catch (e) { return null; }
  }

  async function storeWipe() {
    lsPut('score', { wins: 0, losses: 0, draws: 0, games: 0 });
    lsPut('games', []);
    if (STORE.ok === false) return;
    try {
      var games = await storeGames();
      if (!games) return;
      for (var i = 0; i < games.length; i++) {
        await storeFetch('/v1/apps/' + STORE.app + '/games?game_id=' +
          encodeURIComponent(games[i].game_id), { method: 'DELETE' });
      }
    } catch (e) {}
  }

  function esc(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
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
  function now() { return Date.now(); }
  function uid() { return now().toString(36) + Math.random().toString(36).slice(2, 7); }
  function fmtTime(at) {
    try { return new Date(at).toLocaleTimeString('zh-CN', { hour12: false }); }
    catch (e) { return ''; }
  }

  /* ---------------------------------------------------------------- */
  /* same-origin API                                                    */
  /* ---------------------------------------------------------------- */
  /* No configurable endpoint: the session cookie is the only credential
     and /v1/systemone always belongs to the resident local checkpoint. */
  function apiBase() { return location.origin + '/v1'; }
  function apiUrl(path) { return apiBase() + path; }
  function apiHeaders(json) {
    var h = {};
    if (json) h['Content-Type'] = 'application/json';
    return h;
  }
  function setModelLabel() {
    var el = $('modelLabel');
    if (!el) return;
    /* The family is the part that changes with a model switch: two checkpoints
     * answer through different prompt grammars, and the endpoint says which
     * one it used on every response. */
    var fam = S.adapter && S.adapter.family ? ' · ' + S.adapter.family : '';
    el.textContent = '模型 ' + (S.modelId || '—') + fam;
  }
  function resolveModel() {
    fetch(apiUrl('/models'), { headers: apiHeaders(false), cache: 'no-store' })
      .then(function (r) { if (authRedirect(r)) return null; return r.ok ? r.json() : null; })
      .then(function (j) {
        if (!j || !Array.isArray(j.data)) return;
        var pick = null;
        for (var i = 0; i < j.data.length; i++) {
          if (j.data[i].cwick && j.data[i].cwick.loaded) { pick = j.data[i]; break; }
        }
        S.modelId = (pick && pick.id) || (j.data[0] && j.data[0].id) || '';
        setModelLabel();
      })
      .catch(function () { setModelLabel(); });
  }

  /* A model id cached before a switch on the server is refused by name, on
     every call, forever. Forget it, rediscover the resident one and hand it
     back so the caller can retry once -- the old name must not outlive the
     switch. */
  function rediscoverModel() {
    S.modelId = '';
    return fetch(apiUrl('/models'), { headers: apiHeaders(false), cache: 'no-store' })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (j) {
        if (!j || !Array.isArray(j.data)) return '';
        var pick = null;
        for (var i = 0; i < j.data.length; i++) {
          if (j.data[i].cwick && j.data[i].cwick.loaded) { pick = j.data[i]; break; }
        }
        S.modelId = (pick && pick.id) || (j.data[0] && j.data[0].id) || '';
        setModelLabel();
        return S.modelId;
      })
      .catch(function () { return ''; });
  }

  /* One decision, read out of a single prefill: the model never writes a
     move, it puts a probability on each empty cell and the answer is the
     argmax. That takes the parse step -- and the retry when a model wraps its
     JSON in prose -- off the critical path of a turn. */
  async function askDecision(state, instructions, criteria) {
    var ctrl = new AbortController();
    S.abort = ctrl;
    var t0 = now();
    var res = null, text = '', retried = false;
    for (;;) {
      try {
        res = await fetch(apiUrl('/systemone'), {
          method: 'POST',
          headers: apiHeaders(true),
          signal: ctrl.signal,
          body: JSON.stringify({
            /* An empty name is omitted on purpose: the endpoint then answers
               with the resident model instead of a literal from this file. */
            model: S.modelId || undefined,
            state: state,
            questions: {
              move: { type: 'choice', instructions: instructions, criteria: criteria }
            }
          })
        });
      } catch (e) {
        if (e && e.name === 'AbortError') throw e;
        throw new Error('无法连接 ' + apiBase() + '：' + (e && e.message ? e.message : e));
      }
      if (authRedirect(res)) throw new Error('authentication required');
      if (res.ok) break;
      text = '';
      try { text = await res.text(); } catch (e2) {}
      if (!retried && res.status === 400 && /not served by this endpoint/i.test(text)) {
        retried = true;
        await rediscoverModel();
        continue;
      }
      throw new Error('HTTP ' + res.status + '：' + text.slice(0, 300));
    }
    var j = await res.json();
    var a = (j.answers && j.answers.move) || {};
    if (!a.choice) throw new Error('决策响应里没有 move');
    if (j.model) S.modelId = String(j.model);
    if (j.adapter) { S.adapter = j.adapter; setModelLabel(); }
    return {
      choice: String(a.choice),
      probability: (a.probabilities && a.probabilities[a.choice]) || 0,
      confidence: Number(a.confidence) || 0,
      probabilities: a.probabilities || {},
      usage: j.usage || null,
      adapter: j.adapter || null,
      latency: (now() - t0) / 1000
    };
  }

  /* Board facts, not a move. The goal asks for the cell with the GREATEST
     PROBABILITY OF WINNING, so the app supplies the facts a player would count
     -- an immediate win, a forced block, how many winning replies a placement
     leaves, and how many lines are still open -- and leaves the judgement to
     the checkpoint. Nothing here ranks or picks a cell: the option letters and
     their probability distribution come from one /v1/systemone prefill.
     Keeping the facts out of the old instructions matters: "win, else block,
     else centre, else corner" was a local rule pretending to be a question. */
  function isWin(b, mark) {
    for (var i = 0; i < LINES.length; i++) {
      var l = LINES[i];
      if (b[l[0]] === mark && b[l[1]] === mark && b[l[2]] === mark) return true;
    }
    return false;
  }
  function cellFacts(mark, cell, board) {
    var src = board || S.board;
    var other = mark === 'O' ? 'X' : 'O';
    var out = { win: false, block: false, forks: 0, open: 0, other: other };
    if (src[cell]) return out;
    var b = src.slice();
    b[cell] = mark;
    out.win = isWin(b, mark);
    /* how many distinct cells would win on the next turn after this one */
    for (var i = 0; i < 9; i++) {
      if (b[i]) continue;
      b[i] = mark;
      if (isWin(b, mark)) out.forks++;
      b[i] = '';
    }
    for (var j = 0; j < LINES.length; j++) {
      var l = LINES[j], closed = false;
      for (var k = 0; k < 3; k++) if (src[l[k]] === other) closed = true;
      if (l.indexOf(cell) >= 0 && !closed) out.open++;
    }
    var opp = src.slice();
    opp[cell] = other;
    out.block = isWin(opp, other);
    return out;
  }
  function factsEn(f) {
    var bits = [];
    if (f.win) bits.push('wins immediately');
    if (f.block) bits.push('blocks ' + f.other + ' immediate win');
    bits.push(f.forks + ' winning ' +
              (f.forks === 1 ? 'reply' : 'replies') + ' next');
    bits.push(f.open + ' open line' + (f.open === 1 ? '' : 's'));
    return bits.join(', ');
  }
  function factsZh(f, mark) {
    var bits = [];
    if (f.win) bits.push(mark + ' 立即成三');
    if (f.block) bits.push('挡住 ' + f.other + ' 的立即成三');
    bits.push('后续 ' + f.forks + ' 个制胜点');
    bits.push(f.open + ' 条可用连线');
    return bits.join(' · ');
  }

  /* What the model judges: the board, whose turn it is, and one line of facts
     per empty cell. The question and the option letters are built by the
     endpoint, so the app states the objective and the facts, never a move. */
  function decisionState() {
    var rows = [];
    for (var r = 0; r < 3; r++) {
      var cells = [];
      for (var c = 0; c < 3; c++) {
        var i = r * 3 + c;
        cells.push(i + ':' + (S.board[i] || '.'));
      }
      rows.push(cells.join('  '));
    }
    var mv = [];
    for (var k = 0; k < S.moves.length; k++) mv.push(S.moves[k].mark + ':' + S.moves[k].cell);
    var notes = [];
    for (var n = 0; n < 9; n++)
      if (!S.board[n]) notes.push(n + ': ' + factsEn(cellFacts('O', n)));
    return 'Tic-tac-toe board, cells 0-8 row-major, "." is empty:\n' +
      rows.join('\n') +
      '\nMoves so far: ' + (mv.length ? mv.join(',') : 'none') +
      '\nO moves next; X is the human and played first.' +
      '\nTactical notes for O (facts only):\n' + notes.join('\n');
  }

  var WIN_QUESTION =
    'Which empty cell gives O the greatest probability of winning this game? ' +
    'Judge from the board and the tactical notes: an immediate win, a forced ' +
    'block, future winning replies and open lines all change that probability. ' +
    'Pick the cell with the highest win probability.';

  function cellLabel(i) {
    return 'row ' + (Math.floor(i / 3) + 1) + ' col ' + (i % 3 + 1);
  }


  /* ---------------------------------------------------------------- */
  /* persistence                                                        */
  /* ---------------------------------------------------------------- */
  function loadGame() {
    try {
      var j = JSON.parse(localStorage.getItem(LS_GAME) || '{}') || {};
      if (j.score && typeof j.score === 'object') {
        S.score.wins = Number(j.score.wins) || 0;
        S.score.draws = Number(j.score.draws) || 0;
        S.score.losses = Number(j.score.losses) || 0;
        S.score.games = Number(j.score.games) || 0;
      }
      if (Array.isArray(j.games)) S.games = j.games.slice(-20);
    } catch (e) { /* private mode: a fresh scoreboard is fine */ }
    scoreUI();
  }

  /* The server is authoritative: the scoreboard and the finished games have
   * to be the same ones on the phone that scanned the QR. What localStorage
   * holds is the offline mirror, so it is read first (instant) and then
   * corrected once the server answers. */
  async function loadFromStore() {
    var ok = await storeProbe();
    if (!ok) { renderHistory(); return; }
    var sc = await storeGet('score');
    if (sc && typeof sc === 'object') {
      S.score.wins = Number(sc.wins) || 0;
      S.score.draws = Number(sc.draws) || 0;
      S.score.losses = Number(sc.losses) || 0;
      S.score.games = Number(sc.games) || 0;
      scoreUI();
    }
    await refreshHistory();
  }

  async function refreshHistory() {
    var games = await storeGames();
    if (games) S.history = games;
    renderHistory();
  }

  function saveGame() {
    try {
      localStorage.setItem(LS_GAME, JSON.stringify({
        score: S.score,
        games: S.games.slice(-20)
      }));
    } catch (e) {}
    scoreUI();
    /* Fire and forget: a lost scoreboard write must never stall a move. */
    if (STORE.ok !== false) storePut('score', S.score);
  }
  function scoreUI() {
    if ($('win')) $('win').textContent = String(S.score.wins);
    if ($('draw')) $('draw').textContent = String(S.score.draws);
    if ($('loss')) $('loss').textContent = String(S.score.losses);
  }

  /* ---------------------------------------------------------------- */
  /* game                                                               */
  /* ---------------------------------------------------------------- */
  function emptyCells() {
    var out = [];
    for (var i = 0; i < 9; i++) if (!S.board[i]) out.push(i);
    return out;
  }
  function check() {
    for (var i = 0; i < LINES.length; i++) {
      var l = LINES[i];
      var a = S.board[l[0]], b = S.board[l[1]], c = S.board[l[2]];
      if (a && a === b && b === c) return { winner: a, line: l };
    }
    for (var k = 0; k < 9; k++) if (!S.board[k]) return null;
    return { winner: 'draw', line: null };
  }
  function setStatus(text) { if ($('status')) $('status').textContent = text; }
  function setTurn(label, cls) {
    if ($('turn')) $('turn').innerHTML = '<i class="dot ' + (cls || '') +
      '"></i><span>' + esc(label) + '</span>';
  }

  function renderBoard() {
    var box = $('board');
    if (!box) return;
    var html = '';
    for (var i = 0; i < 9; i++) {
      var m = S.board[i], cls = 'cell';
      if (m === 'X') cls += ' x';
      if (m === 'O') cls += ' o';
      if (S.line && S.line.indexOf(i) >= 0) cls += ' win';
      var disabled = S.over || S.busy || !!m || S.turn !== 'X';
      html += '<button type="button" class="' + cls + '" data-cell="' + i + '"' +
        (disabled ? ' disabled' : '') + ' role="gridcell" aria-label="格 ' + i +
        (m ? '，' + m : '，空') + '"><span class="idx">' + i + '</span>' +
        (m || '') + '</button>';
    }
    box.innerHTML = html;
    if ($('count')) $('count').textContent = String(S.log.length);
  }

  function renderLog() {
    var box = $('log');
    if (!box) return;
    if (!S.log.length && !S.busy) {
      box.innerHTML = '<div class="muted empty">开局后，这里会记录每一步：你落子形成的威胁、' +
        '发给模型的棋盘与各格战术事实、模型对每个空格的胜率读数、实际落子、耗时与兜底判断。</div>';
      return;
    }
    var html = '';
    var start = Math.max(0, S.log.length - 40);
    for (var i = start; i < S.log.length; i++) {
      var e = S.log[i];
      if (e.who === 'game') {
        html += '<div class="ev ' + (e.tone === 'bad' ? 'err' : 'ai') + '">' +
          '<div class="ev-head"><b>' + esc(e.text) + '</b>' +
          '<span class="spacer"></span><span class="mono muted">' +
          fmtTime(e.at) + '</span></div></div>';
        continue;
      }
      var isAi = e.who === 'O';
      var cls = isAi ? 'ai' : 'human';
      if (e.fallback) cls += ' warn';
      if (e.error) cls += ' err';
      html += '<div class="ev ' + cls + '"><div class="ev-head">' +
        '<span class="badge ' + (isAi ? 'o' : 'x') + '">' + e.who + '</span>' +
        '<b>' + (isAi ? '机器落子' : '你落子') + '</b>' +
        '<span class="cellref">#' + (e.cell == null ? '—' : e.cell) + '</span>' +
        '<span class="spacer"></span><span class="mono muted">' +
        fmtTime(e.at) + '</span></div>';
      if (!isAi && e.facts) {
        html += '<div class="reason">' + esc(factsZh(e.facts, 'X')) + '</div>';
      }
      if (e.probs && Object.keys(e.probs).length) {
        /* The readout is a softmax over the option letters of ONE prefill.
           It is the model's distribution over "which cell wins most", not a
           rollout win rate -- say so in the heading instead of implying a
           simulation happened. */
        var keys = Object.keys(e.probs), topKey = '', topP = -1;
        for (var ki = 0; ki < keys.length; ki++) {
          var pv = Number(e.probs[keys[ki]]) || 0;
          if (pv > topP) { topP = pv; topKey = keys[ki]; }
        }
        var probRows = '';
        for (var kj = 0; kj < keys.length; kj++) {
          var k = keys[kj], v = Number(e.probs[k]) || 0;
          probRows += '<div class="prob-row' +
            (k === String(e.cell) ? ' is-pick' : '') + '">' +
            '<span class="prob-cell mono">' + esc(k) + '</span>' +
            '<span class="prob-track"><i style="width:' +
            Math.max(0, Math.min(100, v * 100)).toFixed(1) + '%"></i></span>' +
            '<b class="mono">' + (v * 100).toFixed(1) + '%</b></div>';
        }
        html += '<div class="prob"><div class="prob-head">各空格「赢面最大」的概率读数' +
          ' <span class="muted">· 一次 prefill 的选项分布，非模拟胜率</span></div>' +
          probRows + '<div class="prob-foot">模型读数最高：<b class="mono">#' +
          esc(topKey) + '</b>（' + (topP * 100).toFixed(1) + '%）· 实际落子 <b>#' +
          esc(e.cell == null ? '—' : e.cell) + '</b></div></div>';
      }
      if (isAi) {
        if (e.reason) html += '<div class="reason">' + esc(e.reason) + '</div>';
        if (e.fallback) {
          html += '<div class="meta" style="color:var(--amber)">本地兜底' +
            (e.error ? '：' + esc(e.error) : '') + '</div>';
        }
        var meta = [];
        if (e.latency) meta.push(e.latency.toFixed(1) + ' s');
        if (e.usage) {
          if (e.usage.prompt_tokens != null) meta.push('prompt ' + e.usage.prompt_tokens + ' tok');
          if (e.usage.completion_tokens != null) meta.push('completion ' + e.usage.completion_tokens + ' tok');
        }
        meta.push('effort low');
        html += '<div class="meta mono">' + esc(meta.join(' · ')) + '</div>';
        if (e.prompt || e.content || e.reasoning) {
          html += '<details><summary>查看请求与模型原文</summary><pre>' +
            esc('USER\n' + (e.prompt || '') + '\n\nMODEL CONTENT\n' + (e.content || '') +
                (e.reasoning ? '\n\nMODEL REASONING\n' + e.reasoning : '')) +
            '</pre></details>';
        }
      }
      html += '</div>';
    }
    if (S.busy) {
      var sec = ((now() - S.pendingStart) / 1000).toFixed(1);
      html += '<div class="ev ai pending"><div class="ev-head">' +
        '<span class="badge o">O</span><b>机器思考中</b>' +
        '<span class="spacer"></span><span class="mono muted">' + sec + ' s</span>' +
        '</div><div class="reason"><span class="spin"></span> 已发送棋盘与各空格战术事实，' +
        '等待模型读出「赢面最大」的那一格…' +
        '</div><div class="meta mono">prefill 可能需要 1–3 分钟</div></div>';
    }
    box.innerHTML = html;
    box.scrollTop = box.scrollHeight;
  }

  /* ---------------------------------------------------------------- */
  /* history: the server's finished games, replayed step by step       */
  /* ---------------------------------------------------------------- */
  function miniBoard(board) {
    var h = '<div class="mini" aria-hidden="true">';
    for (var i = 0; i < 9; i++) {
      var m = (board && board[i]) || '';
      h += '<span class="mcell' + (m === 'X' ? ' x' : m === 'O' ? ' o' : '') + '">' +
        (m || '·') + '</span>';
    }
    return h + '</div>';
  }

  function outcomeOf(g) {
    if (g.outcome === 'human') return { text: '你赢', cls: 'ok' };
    if (g.outcome === 'machine') return { text: '机器赢', cls: 'bad' };
    if (g.outcome === 'draw') return { text: '平局', cls: '' };
    return { text: '未结束', cls: 'muted' };
  }

  function replayHtml(events) {
    if (!events.length) return '<div class="muted empty">这局没有记录到步骤。</div>';
    var html = '';
    for (var i = 0; i < events.length; i++) {
      var p = events[i].payload || {};
      var isAi = events[i].kind === 'ai';
      html += '<div class="ev ' + (isAi ? 'ai' : 'human') +
        (p.fallback ? ' warn' : '') + '">' +
        '<div class="ev-head"><span class="badge ' + (isAi ? 'o' : 'x') + '">' +
        (isAi ? 'O' : 'X') + '</span><b>' + (isAi ? '机器落子' : '你落子') + '</b>' +
        '<span class="cellref">#' + (p.cell == null ? '—' : p.cell) + '</span>' +
        '<span class="spacer"></span><span class="mono muted">' +
        fmtTime(events[i].ts * 1000) + '</span></div>' + miniBoard(p.board);
      if (isAi) {
        if (p.reason) html += '<div class="reason">' + esc(p.reason) + '</div>';
        if (p.fallback) {
          html += '<div class="meta" style="color:var(--amber)">本地兜底' +
            (p.error ? '：' + esc(p.error) : '') + '</div>';
        }
        var meta = [];
        if (p.latency) meta.push(Number(p.latency).toFixed(1) + ' s');
        if (p.usage) {
          if (p.usage.prompt_tokens != null) meta.push('prompt ' + p.usage.prompt_tokens + ' tok');
          if (p.usage.completion_tokens != null) meta.push('completion ' + p.usage.completion_tokens + ' tok');
        }
        if (p.model) meta.push(String(p.model));
        if (meta.length) html += '<div class="meta mono">' + esc(meta.join(' · ')) + '</div>';
        if (p.prompt || p.content) {
          html += '<details><summary>查看请求与模型原文</summary><pre>' +
            esc('USER\n' + (p.prompt || '') + '\n\nMODEL CONTENT\n' + (p.content || '') +
                (p.reasoning ? '\n\nMODEL REASONING\n' + p.reasoning : '')) +
            '</pre></details>';
        }
      }
      html += '</div>';
    }
    return html;
  }

  function renderHistory() {
    var box = $('history');
    if (!box) return;
    if (!S.history.length) {
      box.innerHTML = '<div class="muted empty">' +
        (STORE.ok === false
          ? '服务端存储不可用，历史只保存在本机浏览器（localStorage）。'
          : '还没有已结束的对局。下完一局后，这里会列出它，点开可以逐步回放。') +
        '</div>';
      return;
    }
    var html = '<div class="hist-head"><span class="muted">最近 ' +
      S.history.length + ' 局</span>' +
      (STORE.ok === false ? '' : '<span class="muted mono">sqlite</span>') +
      '</div>';
    for (var i = 0; i < S.history.length; i++) {
      var g = S.history[i];
      var o = outcomeOf(g);
      html += '<button type="button" class="hist-item' +
        (S.replay && S.replay.id === g.game_id ? ' is-open' : '') +
        '" data-game="' + esc(g.game_id) + '">' +
        '<span class="mono muted">' + fmtTime((g.started_at || 0) * 1000) + '</span>' +
        '<b class="' + o.cls + '">' + o.text + '</b>' +
        '<span class="mono muted">' + (g.moves || 0) + ' 手</span></button>';
    }
    if (S.replay) {
      html += '<div class="replay">' +
        (S.replay.loading ? '<div class="muted">正在读取这一局的步骤…</div>'
                          : replayHtml(S.replay.events)) + '</div>';
    }
    box.innerHTML = html;
  }

  async function openReplay(gameId) {
    if (S.replay && S.replay.id === gameId && !S.replay.loading) {
      S.replay = null;                 /* second click collapses it */
      renderHistory();
      return;
    }
    S.replay = { id: gameId, events: [], loading: true };
    renderHistory();
    var evs = await storeEvents(gameId);
    S.replay = { id: gameId, events: evs || [], loading: false };
    renderHistory();
  }

  function setSideTab(tab) {
    S.sideTab = tab;
    var now = $('tabNow'), hist = $('tabHistory');
    if (now) now.classList.toggle('is-active', tab === 'now');
    if (hist) hist.classList.toggle('is-active', tab === 'history');
    if ($('log')) $('log').hidden = tab !== 'now';
    if ($('history')) $('history').hidden = tab !== 'history';
    if (tab === 'history') renderHistory();
  }

  function fallbackMove() {
    var empty = emptyCells();
    if (!empty.length) return -1;
    function winFor(mark) {
      for (var i = 0; i < LINES.length; i++) {
        var l = LINES[i], n = 0, at = -1;
        for (var k = 0; k < 3; k++) {
          if (S.board[l[k]] === mark) n++;
          else if (!S.board[l[k]]) at = l[k];
        }
        if (n === 2 && at >= 0) return at;
      }
      return -1;
    }
    var w = winFor('O'); if (w >= 0) return w;
    var b = winFor('X'); if (b >= 0) return b;
    if (!S.board[4]) return 4;
    var corners = [0, 2, 6, 8];
    for (var i = 0; i < corners.length; i++) if (!S.board[corners[i]]) return corners[i];
    return empty[0];
  }

  function newGame() {
    if (S.abort) { try { S.abort.abort(); } catch (e) {} }
    if (S.timer) { clearInterval(S.timer); S.timer = 0; }
    S.gen++;
    S.board = ['', '', '', '', '', '', '', '', ''];
    S.turn = 'X';
    S.over = false;
    S.winner = null;
    S.line = null;
    S.busy = false;
    S.moves = [];
    S.log = [];
    S.startedAt = now();
    S.gameId = uid();
    S.seq = 0;
    S.started = false;
    S.abort = null;
    setStatus('你执 X，先手。点击棋盘落子。');
    setTurn('轮到你（X）', '');
    renderBoard(); renderLog(); saveGame();
  }

  /* The game row is created on the first move, not on "new game": otherwise
   * every page reload would leave behind an abandoned match in the history. */
  function ensureGame() {
    if (S.started) return;
    S.started = true;
    if (STORE.ok !== false) {
      storeGamePut({
        game_id: S.gameId,
        started_at: Math.floor(S.startedAt / 1000)
      });
    }
  }

  /* One row per step, carrying everything the replay needs: the board after
   * the move, what was sent to the model, what came back, and how long it
   * took. The board snapshot is what makes a replay readable without
   * re-deriving it from the move list. */
  function recordMove(kind, entry) {
    if (STORE.ok === false) return;
    storeEventAdd(S.gameId, S.seq++, kind, entry);
  }

  function finish(chk) {
    if (S.over) return;
    S.over = true;
    S.winner = chk.winner;
    S.line = chk.line;
    var result = 'draw', text = '平局：棋盘已满';
    if (chk.winner === 'X') {
      S.score.wins++;
      result = 'human'; text = '你赢了！X 连成一线';
      S.log.push({ who: 'game', text: text, tone: 'ok', at: now() });
      setStatus('你赢了。点击“新一局”再来一盘。');
      setTurn('你赢了（X）', '');
    } else if (chk.winner === 'O') {
      S.score.losses++;
      result = 'machine'; text = '机器赢了：O 连成一线';
      S.log.push({ who: 'game', text: text, tone: 'bad', at: now() });
      setStatus('机器赢了。点击“新一局”再试一次。');
      setTurn('机器赢了（O）', 'bad');
    } else {
      S.score.draws++;
      S.log.push({ who: 'game', text: text, tone: 'ok', at: now() });
      setStatus('平局。点击“新一局”继续。');
      setTurn('平局', '');
    }
    S.score.games++;
    S.games.push({
      id: S.gameId, startedAt: S.startedAt, finishedAt: now(), result: result,
      moves: S.moves.slice(), board: S.board.slice(), model: S.modelId,
      decisions: S.log.filter(function (e) { return e.who === 'O'; }).map(function (e) {
        return { cell: e.cell, reason: e.reason, latency: e.latency,
                 fallback: !!e.fallback, error: e.error || '' };
      })
    });
    if (S.games.length > 20) S.games = S.games.slice(-20);
    saveGame();
    /* Close the server's game row, then pull the history: reading it in
     * parallel would list this match as still running and with 0 moves. */
    if (STORE.ok !== false) {
      storeGamePut({
        game_id: S.gameId,
        started_at: Math.floor(S.startedAt / 1000),
        ended_at: Math.floor(now() / 1000),
        outcome: result,
        moves: S.moves.length
      }).then(function () { refreshHistory(); });
    }
  }

  function human(cell) {
    if (S.busy || S.over || S.turn !== 'X' || S.board[cell]) return;
    var facts = cellFacts('X', cell);      /* before the stone is on the board */
    S.board[cell] = 'X';
    S.moves.push({ mark: 'X', cell: cell });
    S.log.push({ who: 'X', cell: cell, at: now(), facts: facts });
    ensureGame();
    recordMove('human', { cell: cell, board: S.board.slice(), at: now() });
    renderBoard(); renderLog(); saveGame();
    var chk = check();
    if (chk) { finish(chk); renderBoard(); renderLog(); return; }
    S.turn = 'O';
    renderBoard();
    aiTurn();
  }

  async function aiTurn() {
    var gen = S.gen;
    S.busy = true;
    S.pendingStart = now();
    renderBoard(); renderLog();
    setTurn('机器（O）思考中', 'wait');
    setStatus('机器正在通过 /v1/systemone 决策…');
    if (S.timer) clearInterval(S.timer);
    S.timer = setInterval(function () {
      if (gen !== S.gen) return;
      var sec = ((now() - S.pendingStart) / 1000).toFixed(0);
      setStatus('机器正在通过 /v1/systemone 决策… ' + sec + 's');
      renderLog();
    }, 500);

    var state = decisionState();
    var empties = emptyCells();
    var crit = {};
    for (var ci = 0; ci < empties.length; ci++) {
      var cell = empties[ci];
      crit[String(cell)] = cellLabel(cell) + ' — ' + factsEn(cellFacts('O', cell));
    }
    var entry = {
      who: 'O', cell: null, at: now(), latency: 0, reason: '',
      fallback: false, error: '', content: '', reasoning: '',
      usage: null, prompt: state
    };
    var move = null;

    try {
      var r = await askDecision(state, WIN_QUESTION, crit);
      if (gen !== S.gen) return;
      entry.latency = r.latency;
      entry.usage = r.usage;
      entry.probs = r.probabilities;
      entry.content = JSON.stringify(r.probabilities);
      var parsed = Number(r.choice);
      if (isFinite(parsed) && S.board[parsed] === '' && parsed >= 0 && parsed < 9) {
        move = parsed;
        entry.reason = '模型给 ' + cellLabel(move) + ' 的概率 ' +
          (r.probability * 100).toFixed(1) + '%，置信度 ' +
          (r.confidence * 100).toFixed(1) + '%。';
      } else {
        move = fallbackMove();
        entry.fallback = true;
        entry.error = '决策响应里的格子在棋盘上不可用（已回退到本地规则）';
        entry.reason = '模型选择不可用，按取胜 / 拦截 / 中心 / 角 / 边顺序本地兜底。';
      }
    } catch (e) {
      if (gen !== S.gen) return;
      if (e && e.name === 'AbortError') {
        S.busy = false;
        if (S.timer) { clearInterval(S.timer); S.timer = 0; }
        S.abort = null;
        setTurn('已取消', 'bad');
        setStatus('已取消本手。点击“新一局”重新开始。');
        renderLog();
        return;
      }
      move = fallbackMove();
      entry.fallback = true;
      entry.error = (e && e.message) ? e.message : String(e);
      entry.latency = (now() - S.pendingStart) / 1000;
      entry.reason = '请求失败，按本地规则兜底；下一手会继续尝试模型。';
    }

    if (gen !== S.gen) return;
    S.busy = false;
    if (S.timer) { clearInterval(S.timer); S.timer = 0; }
    S.abort = null;

    if (move == null || move < 0) {
      S.turn = 'X';
      renderBoard(); renderLog();
      finish({ winner: 'draw', line: null });
      renderBoard(); renderLog();
      return;
    }
    entry.cell = move;
    S.log.push(entry);
    S.board[move] = 'O';
    S.moves.push({ mark: 'O', cell: move });
    S.turn = 'X';
    recordMove('ai', {
      cell: move, board: S.board.slice(), reason: entry.reason,
      latency: entry.latency, fallback: !!entry.fallback,
      error: entry.error || '', prompt: entry.prompt,
      content: entry.content, reasoning: entry.reasoning, usage: entry.usage,
      probs: entry.probs || {}, model: S.modelId
    });
    var chk = check();
    renderBoard(); renderLog(); saveGame();
    if (chk) { finish(chk); renderBoard(); renderLog(); return; }
    setTurn('轮到你（X）', '');
    setStatus('轮到你落子（X）。');
  }

  async function resetScore() {
    await storeWipe();
    S.score = { wins: 0, losses: 0, draws: 0, games: 0 };
    S.games = [];
    S.history = [];
    S.replay = null;
    scoreUI();
    renderHistory();
    setStatus('战绩与服务端历史已清空。');
  }


  /* ---------------------------------------------------------------- */
  /* share / QR                                                         */
  /* ---------------------------------------------------------------- */
  async function share() {
    var dlg = $('shareDialog'), img = $('qrImg'), urlEl = $('shareUrl'), note = $('shareNote');
    if (!dlg || !img || !urlEl || !note) return;
    img.hidden = true;
    img.removeAttribute('src');
    urlEl.textContent = '正在读取服务地址…';
    note.className = 'share-note';
    note.textContent = '';
    if (dlg.showModal) dlg.showModal(); else dlg.setAttribute('open', '');
    try {
      var res = await fetch('/v1/share?app=tictactoe', { cache: 'no-store' });
      if (authRedirect(res)) return;
      if (!res.ok) throw new Error('HTTP ' + res.status);
      var j = await res.json();
      var url = j.app_url ||
        ((j.lan_urls && j.lan_urls.length) ? j.lan_urls[0] + '/apps/tictactoe/index.html' : '');
      if (!j.shareable || !url) {
        urlEl.textContent = location.origin + '/apps/tictactoe/index.html';
        note.className = 'share-note bad';
        note.textContent = (j.warning || '当前服务没有可分享的局域网地址。') +
          ' 在这台 Mac 上运行 ./start.sh --lan（或 cwick serve --lan）后重试。';
        return;
      }
      urlEl.textContent = url;
      img.src = '/v1/share/qr?text=' + encodeURIComponent(url);
      img.hidden = false;
      note.textContent = '手机与这台 Mac 连同一个 Wi-Fi，扫码后用浏览器打开即可独立运行；首次打开需要输入这台 Mac 的登录密码。';
    } catch (e) {
      urlEl.textContent = location.origin + '/apps/tictactoe/index.html';
      note.className = 'share-note bad';
      note.textContent = '无法读取 /v1/share：' + (e && e.message ? e.message : e);
    }
  }

  function closeShare() {
    var dlg = $('shareDialog');
    if (!dlg) return;
    if (dlg.close) dlg.close(); else dlg.removeAttribute('open');
  }

  async function copyShare() {
    var el = $('shareUrl'); if (!el) return;
    var text = el.textContent || '';
    try {
      if (navigator.clipboard) await navigator.clipboard.writeText(text);
      else {
        var t = document.createElement('textarea');
        t.value = text; document.body.appendChild(t); t.select();
        document.execCommand('copy'); t.remove();
      }
      var b = $('copyShare');
      if (b) { b.textContent = '已复制'; setTimeout(function () { b.textContent = '复制链接'; }, 1200); }
    } catch (e) {}
  }

  function bind() {
    if ($('board')) $('board').addEventListener('click', function (e) {
      var cell = e.target.closest('.cell');
      if (!cell || cell.disabled) return;
      human(Number(cell.dataset.cell));
    });
    if ($('newGame')) $('newGame').addEventListener('click', newGame);
    if ($('share')) $('share').addEventListener('click', share);
    if ($('shareClose')) $('shareClose').addEventListener('click', closeShare);
    if ($('copyShare')) $('copyShare').addEventListener('click', copyShare);
    if ($('resetScore')) $('resetScore').addEventListener('click', function () {
      if (window.confirm('清空战绩与服务端的对局历史（含每一步的回放）？')) resetScore();
    });
    if ($('tabNow')) $('tabNow').addEventListener('click', function () { setSideTab('now'); });
    if ($('tabHistory')) $('tabHistory').addEventListener('click', function () { setSideTab('history'); });
    if ($('clearHistory')) $('clearHistory').addEventListener('click', function () {
      if (window.confirm('清空战绩与服务端的对局历史（含每一步的回放）？')) resetScore();
    });
    if ($('history')) $('history').addEventListener('click', function (e) {
      var item = e.target.closest('.hist-item');
      if (!item) return;
      openReplay(item.dataset.game);
    });
  }

  function boot() {
    if (location.search.indexOf('embed=1') >= 0) document.body.classList.add('embed');
    loadGame();
    bind();
    resolveModel();
    newGame();
    setSideTab('now');
    /* After the board exists: the storage badge and the history come from the
     * server, and a slow answer must not delay the first move. */
    loadFromStore();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
