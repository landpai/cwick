/* 3D 泡泡卡丁车 — a lap-running game on a Möbius strip whose obstacles are
 * decided, one at a time, by the resident model.
 *
 * Scope, deliberately: the model has exactly one job here — say where the next
 * obstacle goes. There is no local generator behind it and no second AI feature:
 * if the model does not answer, nothing is placed on the strip until it does.
 * Randomness lives in the *question*, not in the road: each real-time decision
 * shuffles the twelve option letters, draws a variation seed and a different
 * pressure point, and asks the resident checkpoint to make the lap as hard as
 * it can. The checkpoint's prefill is still the only thing that chooses an
 * obstacle; the app never substitutes a locally generated one. That is what
 * makes the sidebar honest — every obstacle has a model decision behind it.
 *
 * Rendering is a 2D canvas with a painter-sorted triangle buffer (~620 tris a
 * frame). No WebGL, no library: the app is served as a static asset from the
 * same origin as the API, has to run on a phone that scanned a QR, and has no
 * build step. The track is a real Möbius strip,
 *     S(u, v) = C(u) + A(u)·v,  A(u) = cos(u/2)·radial(u) + sin(u/2)·up
 * so the across direction reverses after one lap and the surface has one side.
 * The kart's frame is sampled from that surface, which is why the horizon rolls
 * as you drive: the camera stays above the surface, not above the world.
 */
(function () {
  'use strict';

  /* ------------------------------------------------------------------ */
  /* config                                                              */
  /* ------------------------------------------------------------------ */
  var R = 72;                 /* centerline radius, metres */
  var W = 4.2;                /* half width of the strip */
  var LIM = W - 0.95;         /* how far the kart may drift sideways */
  var NEAR = 0.35;
  var FOV = 1.12;             /* ~64 degrees */
  var FOG_START = 70, FOG_END = 420;
  var MAX_SPEED = 26;         /* m/s before per-lap gains */
  var KMH = 3.6;
  var TWO_PI = Math.PI * 2;
  var LS_KART = 'cwick.kart.v3';

  var TYPES = ['block', 'cone', 'duo', 'roll'];
  var TYPE_NAME = { block: '方框路障', cone: '锥桶路障', duo: '双连路障', roll: '滚动路障' };
  /* The options are prefilled on every obstacle; keep their wording short and
     single-byte. The Chinese names are for the sidebar, not the model. */
  var TYPE_EN = { block: 'barrier', cone: 'cone', duo: 'double barrier',
                  roll: 'rolling barrel' };
  var TYPE_COLOR = {
    block: { r: 255, g: 111, b: 125 },
    cone:  { r: 245, g: 185, b: 92 },
    duo:   { r: 167, g: 139, b: 250 },
    roll:  { r: 255, g: 127, b: 208 }
  };
  var DIFF_NAME = ['低', '中', '高', '超高', '极致'];

  /* ------------------------------------------------------------------ */
  /* state                                                               */
  /* ------------------------------------------------------------------ */
  var G = {
    mode: 'idle',             /* idle | running | paused */
    u: 0.6, v: 0, speed: 0, lap: 1, dist: 0, score: 0, hits: 0, spawned: 0,
    byType: { block: 0, cone: 0, duo: 0, roll: 0 },
    obstacles: [], plan: [], asking: false, askFail: 0, askAgainIn: 0,
    boost: 0, boostCd: 0, invuln: 0, camU: 0.6, camV: 0,
    lapStart: 0, bestLap: 0, runStart: 0, elapsed: 0,
    gameId: '', startedAt: 0, events: [], evSeq: 0, storeOk: null,
    aiState: 'idle'           /* idle | busy | waiting | down */
  };
  var AGENT = { calls: 0, ok: 0, fail: 0, msTotal: 0, msN: 0, last: null,
                lastSeed: '', lastFocus: '', lastEntropy: 0 };
  var BEST = { score: 0, lap: 1, lapMs: 0 };
  var LIFE = { runs: 0, laps: 0, dist: 0, hits: 0, spawned: 0 };
  var QUALITY = 1;
  /* ?cam=<metres> raises the chase camera for a wider view of the strip. It is
   * how the loop was tuned: at 5 m you are inside the twist, at 9 m the ribbon
   * reads as a ribbon. */
  var CAM_H = (function () {
    var m = /[?&]cam=([0-9.]+)/.exec(location.search);
    return m ? Math.max(3, Math.min(20, Number(m[1]))) : 5.6;
  })();
  var keys = { left: false, right: false };
  var dragSteer = 0;

  var $ = function (id) { return document.getElementById(id); };
  function clamp(x, a, b) { return x < a ? a : x > b ? b : x; }
  function fmt(n, d) { var p = Math.pow(10, d || 0); return (Math.round(n * p) / p).toFixed(d || 0); }
  function secs(ms) {
    if (!ms) return '—';
    var s = ms / 1000;
    return s >= 60 ? (s / 60).toFixed(1) + ' 分' : s.toFixed(2) + ' s';
  }
  function esc(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }
  function wrapDu(a) {
    var d = a % TWO_PI;
    if (d > Math.PI) d -= TWO_PI;
    if (d < -Math.PI) d += TWO_PI;
    return d;
  }

  /* ------------------------------------------------------------------ */
  /* storage: server sqlite first, localStorage as the fallback          */
  /* ------------------------------------------------------------------ */
  function lsRead() {
    try { return JSON.parse(localStorage.getItem(LS_KART) || '{}') || {}; }
    catch (e) { return {}; }
  }
  function lsWrite(patch) {
    try {
      var j = lsRead();
      for (var k in patch) if (Object.prototype.hasOwnProperty.call(patch, k)) j[k] = patch[k];
      localStorage.setItem(LS_KART, JSON.stringify(j));
      return true;
    } catch (e) { return false; }
  }
  function lsGet(k) { var j = lsRead(); return j[k] == null ? null : j[k]; }

  function authRedirect(res) {
    if (!res || res.status !== 401) return false;
    try { location.replace('/login?next=' + encodeURIComponent(location.pathname + location.search)); } catch (e) {}
    return true;
  }
  function storeFetch(path, opts) {
    return fetch(path, Object.assign({ cache: 'no-store' }, opts || {}));
  }
  function putState(key, value) {
    lsWrite((function () { var o = {}; o[key] = value; return o; })());
    if (G.storeOk === false) return Promise.resolve(false);
    return storeFetch('/v1/apps/kart/state', {
      method: 'PUT', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ key: key, value: value })
    }).then(function (r) {
      if (authRedirect(r)) return false;
      if (r.status === 503) G.storeOk = false;
      return r.ok;
    }).catch(function () { G.storeOk = false; return false; });
  }

  /* Events are queued and flushed one at a time, oldest first: a lap race emits
   * a few per minute, but one request per event would still put the game's IO
   * ahead of the model's. */
  var evBuf = [];
  var flushTimer = 0;
  function record(kind, text, payload) {
    var ev = {
      seq: G.evSeq++, t: Math.round(G.elapsed * 100) / 100,
      kind: kind, text: text, lap: G.lap, speed: Math.round(G.speed * KMH)
    };
    if (payload) {
      for (var k in payload) {
        if (Object.prototype.hasOwnProperty.call(payload, k)) ev[k] = payload[k];
      }
    }
    G.events.push(ev);
    if (G.events.length > 400) G.events.shift();
    renderFeeds();
    evBuf.push(ev);
    scheduleFlush(420);
  }
  function scheduleFlush(delay) {
    if (flushTimer) return;
    flushTimer = setTimeout(flushEvents, delay);
  }
  function flushEvents() {
    flushTimer = 0;
    if (!evBuf.length) return;
    if (G.storeOk === false || !G.gameId) { evBuf.length = 0; return; }
    var ev = evBuf.shift();
    storeFetch('/v1/apps/kart/events', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ game_id: G.gameId, seq: ev.seq, kind: ev.kind, payload: eventPayload(ev) })
    }).then(function (r) {
      if (r.status === 503) G.storeOk = false;
      if (evBuf.length) scheduleFlush(90);
    }).catch(function () {
      G.storeOk = false;
      evBuf.length = 0;
    });
  }
  /* The payload keeps what a replay needs, not the rendered HTML. */
  function eventPayload(ev) {
    var out = {};
    ['t', 'text', 'type', 'lane', 'gap', 'level', 'ms', 'why', 'raw',
     'speed', 'score', 'lap_ms', 'seed', 'focus', 'entropy',
     'confidence'].forEach(function (k) {
      if (ev[k] !== undefined) out[k] = ev[k];
    });
    return out;
  }
  function putGame(fields) {
    if (G.storeOk === false || !G.gameId) return Promise.resolve(false);
    return storeFetch('/v1/apps/kart/games', {
      method: 'PUT', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(Object.assign({ game_id: G.gameId }, fields))
    }).then(function (r) {
      if (r.status === 503) G.storeOk = false;
      return r.ok;
    }).catch(function () { G.storeOk = false; return false; });
  }

  /* ------------------------------------------------------------------ */
  /* Möbius geometry                                                     */
  /* ------------------------------------------------------------------ */
  function surf(u, v, out) {
    var cu = Math.cos(u), su = Math.sin(u), h = u * 0.5, ch = Math.cos(h), sh = Math.sin(h);
    out = out || {};
    out.x = R * cu + ch * cu * v;
    out.y = sh * v;
    out.z = R * su + ch * su * v;
    return out;
  }
  function tangent(u, out) {
    out = out || {};
    out.x = -Math.sin(u); out.y = 0; out.z = Math.cos(u);
    return out;
  }
  function across(u, out) {
    var h = u * 0.5, ch = Math.cos(h), sh = Math.sin(h);
    out = out || {};
    out.x = ch * Math.cos(u); out.y = sh; out.z = ch * Math.sin(u);
    return out;
  }
  function normalAt(u, out) {
    var t = tangent(u, _t1), a = across(u, _a1);
    var nx = t.y * a.z - t.z * a.y;
    var ny = t.z * a.x - t.x * a.z;
    var nz = t.x * a.y - t.y * a.x;
    var l = Math.sqrt(nx * nx + ny * ny + nz * nz) || 1;
    out = out || {};
    out.x = nx / l; out.y = ny / l; out.z = nz / l;
    return out;
  }
  function frameAt(u, v) {
    return { p: surf(u, v), f: tangent(u), r: across(u), n: normalAt(u) };
  }
  var _t1 = {}, _a1 = {};

  /* ------------------------------------------------------------------ */
  /* renderer                                                            */
  /* ------------------------------------------------------------------ */
  var cv = $('cv'), ctx = cv.getContext('2d');
  var Wpx = 0, Hpx = 0, DPR = 1, focal = 1;
  var cam = { eye: {}, f: {}, r: {}, u: {} };
  var tris = [];
  var nTri = 0;

  function resize() {
    var box = cv.parentNode.getBoundingClientRect();
    DPR = Math.min(2, window.devicePixelRatio || 1) * QUALITY;
    Wpx = Math.max(320, Math.round(box.width));
    Hpx = Math.max(240, Math.round(box.height));
    cv.width = Math.round(Wpx * DPR);
    cv.height = Math.round(Hpx * DPR);
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    focal = (Hpx * 0.5) / Math.tan(FOV * 0.5);
  }

  function setCamera() {
    /* Chase cam: ~11 m behind, 4 m above the surface. Anchored to the surface
     * normal, so the horizon rolls with the strip. */
    var eu = G.camU - 0.15;
    var eye = surf(eu, G.camV * 0.55);
    var en = normalAt(eu);
    var h = CAM_H + Math.sin(G.dist * 2.4) * 0.06;   /* suspension reaches the camera */
    eye.x += en.x * h; eye.y += en.y * h; eye.z += en.z * h;

    /* Aim a little closer on a narrow screen: the touch row lives at the
     * bottom edge and a chase cam tuned for a laptop puts the kart right
     * underneath it. */
    var tu = G.u + (Wpx < 620 ? 0.20 : 0.32);
    var tgt = surf(tu, G.camV * 0.42);
    var tn = normalAt(tu);
    tgt.x += tn.x * 0.15; tgt.y += tn.y * 0.15; tgt.z += tn.z * 0.15;

    var f = { x: tgt.x - eye.x, y: tgt.y - eye.y, z: tgt.z - eye.z };
    var fl = Math.hypot(f.x, f.y, f.z) || 1;
    f.x /= fl; f.y /= fl; f.z /= fl;

    var r = {
      x: f.y * en.z - f.z * en.y,
      y: f.z * en.x - f.x * en.z,
      z: f.x * en.y - f.y * en.x
    };
    var rl = Math.hypot(r.x, r.y, r.z) || 1;
    r.x /= rl; r.y /= rl; r.z /= rl;
    var u2 = {
      x: r.y * f.z - r.z * f.y,
      y: r.z * f.x - r.x * f.z,
      z: r.x * f.y - r.y * f.x
    };
    cam.eye = eye; cam.f = f; cam.r = r; cam.u = u2;
  }

  function toCam(p) {
    var dx = p.x - cam.eye.x, dy = p.y - cam.eye.y, dz = p.z - cam.eye.z;
    return {
      x: dx * cam.r.x + dy * cam.r.y + dz * cam.r.z,
      y: dx * cam.u.x + dy * cam.u.y + dz * cam.u.z,
      z: dx * cam.f.x + dy * cam.f.y + dz * cam.f.z   /* positive = in front */
    };
  }
  function project(c) {
    return { x: Wpx * 0.5 + focal * c.x / c.z, y: Hpx * 0.5 - focal * c.y / c.z };
  }

  /* Sutherland–Hodgman against the near plane, in camera space: emitting the
   * clipped polygon instead of dropping the triangle is what keeps the track
   * from vanishing when a segment passes behind the camera. */
  function clipNear(poly) {
    var out = [], i, a, b, t;
    for (i = 0; i < poly.length; i++) {
      a = poly[i];
      b = poly[(i + 1) % poly.length];
      var ain = a.z >= NEAR, bin = b.z >= NEAR;
      if (ain) out.push(a);
      if (ain !== bin) {
        t = (NEAR - a.z) / (b.z - a.z);
        out.push({ x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t, z: NEAR });
      }
    }
    return out;
  }

  function pushTri(a, b, c, fill, depth, alpha) {
    var poly = clipNear([a, b, c]);
    if (poly.length < 3) return;
    var pts = [], i;
    for (i = 0; i < poly.length; i++) pts.push(project(poly[i]));
    for (i = 1; i + 1 < pts.length; i++) {
      tris.push({
        z: depth, fill: fill, a: alpha == null ? 1 : alpha,
        x1: pts[0].x, y1: pts[0].y,
        x2: pts[i].x, y2: pts[i].y,
        x3: pts[i + 1].x, y3: pts[i + 1].y
      });
      nTri++;
    }
  }

  var LIGHT = { x: -0.42, y: 0.82, z: -0.38 };
  /* The normal handed in is a cross product, whose length is the area of the
   * quad — not 1. Dotting that with the light and using it as a brightness
   * factor saturates every large quad to white (a road segment is 8.4 m by
   * 1.7 m, so |n| ~ 14), which is how one build turned the whole strip into a
   * white slab. Normalise first, then shade. */
  function shade(base, nx, ny, nz, dist) {
    var nl = Math.sqrt(nx * nx + ny * ny + nz * nz) || 1;
    var d = Math.abs((nx * LIGHT.x + ny * LIGHT.y + nz * LIGHT.z) / nl);
    var k = 0.42 + 0.68 * d;
    var r = base.r * k, g = base.g * k, b = base.b * k;
    if (dist > FOG_START) {
      var f = Math.min(1, (dist - FOG_START) / (FOG_END - FOG_START));
      f *= f;
      r += (10 - r) * f; g += (16 - g) * f; b += (26 - b) * f;
    }
    return 'rgb(' + (r | 0) + ',' + (g | 0) + ',' + (b | 0) + ')';
  }

  function quad(p0, p1, p2, p3, base, alpha) {
    var c0 = toCam(p0), c1 = toCam(p1), c2 = toCam(p2), c3 = toCam(p3);
    var z = (c0.z + c1.z + c2.z + c3.z) * 0.25;
    var nx = (p1.y - p0.y) * (p2.z - p0.z) - (p1.z - p0.z) * (p2.y - p0.y);
    var ny = (p1.z - p0.z) * (p2.x - p0.x) - (p1.x - p0.x) * (p2.z - p0.z);
    var nz = (p1.x - p0.x) * (p2.y - p0.y) - (p1.y - p0.y) * (p2.x - p0.x);
    var fill = shade(base, nx, ny, nz, z);
    pushTri(c0, c1, c2, fill, z, alpha);
    pushTri(c0, c2, c3, fill, z, alpha);
  }
  function tri3(a, b, c, base, alpha) {
    var ca = toCam(a), cb = toCam(b), cc = toCam(c);
    var z = (ca.z + cb.z + cc.z) / 3;
    var nx = (b.y - a.y) * (c.z - a.z) - (b.z - a.z) * (c.y - a.y);
    var ny = (b.z - a.z) * (c.x - a.x) - (b.x - a.x) * (c.z - a.z);
    var nz = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
    pushTri(ca, cb, cc, shade(base, nx, ny, nz, z), z, alpha);
  }

  /* Box in a local frame: r = right, n = up, f = forward. */
  function boxAt(fr, ox, oy, oz, hx, hy, hz, base, alpha) {
    var r = fr.r, n = fr.n, f = fr.f, p = fr.p;
    function corner(sx, sy, sz) {
      return {
        x: p.x + r.x * (ox + sx * hx) + n.x * (oy + sy * hy) + f.x * (oz + sz * hz),
        y: p.y + r.y * (ox + sx * hx) + n.y * (oy + sy * hy) + f.y * (oz + sz * hz),
        z: p.z + r.z * (ox + sx * hx) + n.z * (oy + sy * hy) + f.z * (oz + sz * hz)
      };
    }
    var A = corner(-1, -1, -1), B = corner(1, -1, -1), C = corner(1, 1, -1),
        D = corner(-1, 1, -1), E = corner(-1, -1, 1), F = corner(1, -1, 1),
        Gc = corner(1, 1, 1), H = corner(-1, 1, 1);
    quad(A, B, C, D, base, alpha);
    quad(E, F, Gc, H, base, alpha);
    quad(A, E, H, D, base, alpha);
    quad(B, F, Gc, C, base, alpha);
    quad(A, E, F, B, base, alpha);
    quad(D, H, Gc, C, base, alpha);
  }

  function octa(fr, ox, oy, oz, rad, base, alpha) {
    var r = fr.r, n = fr.n, f = fr.f, p = fr.p;
    function at(sx, sy, sz) {
      return {
        x: p.x + r.x * (ox + sx * rad) + n.x * (oy + sy * rad) + f.x * (oz + sz * rad),
        y: p.y + r.y * (ox + sx * rad) + n.y * (oy + sy * rad) + f.y * (oz + sz * rad),
        z: p.z + r.z * (ox + sx * rad) + n.z * (oy + sy * rad) + f.z * (oz + sz * rad)
      };
    }
    var top = at(0, 1, 0), bot = at(0, -1, 0);
    var e = [at(1, 0, 0), at(0, 0, 1), at(-1, 0, 0), at(0, 0, -1)];
    for (var i = 0; i < 4; i++) {
      tri3(top, e[i], e[(i + 1) % 4], base, alpha);
      tri3(bot, e[(i + 1) % 4], e[i], base, alpha);
    }
  }

  /* ------------------------------------------------------------------ */
  /* scene                                                               */
  /* ------------------------------------------------------------------ */
  function drawSky() {
    var g = ctx.createLinearGradient(0, 0, 0, Hpx);
    g.addColorStop(0, '#050810');
    g.addColorStop(0.5, '#0a1420');
    g.addColorStop(1, '#0f1a26');
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, Wpx, Hpx);
    var rg = ctx.createRadialGradient(Wpx * 0.5, Hpx * 0.6, 8, Wpx * 0.5, Hpx * 0.6, Hpx * 0.9);
    rg.addColorStop(0, 'rgba(86,216,255,0.11)');
    rg.addColorStop(1, 'rgba(5,8,16,0)');
    ctx.fillStyle = rg;
    ctx.fillRect(0, 0, Wpx, Hpx);
  }

  var ROAD_A = { r: 46, g: 58, b: 76 };
  var ROAD_B = { r: 26, g: 34, b: 46 };
  var CURB_A = { r: 86, g: 216, b: 255 };
  var CURB_B = { r: 255, g: 127, b: 208 };
  var CURB_DIM = { r: 62, g: 76, b: 96 };
  var FAR_ROAD = { r: 30, g: 38, b: 50 };
  var RIM = { r: 78, g: 196, b: 232 };

  /* The strip is drawn in two zones. Up close it needs every detail — banding,
   * kerbs, dashes — because that is what speed is read from. Further out only
   * the silhouette matters, and that silhouette is the point of the whole map:
   * a Möbius strip cannot be understood from the ten metres under the kart.
   * So the far zone (out to ~4 rad, 2/3 of the lap) is one coarse quad per
   * 0.075 rad plus its glowing rim — ~40 quads that show the ribbon curving up
   * over the camera, twisting, and coming back. */
  function drawTrack() {
    var edge = 0.6;
    var rows = [-W, -W + edge, W - edge, W];

    /* --- near zone: world-aligned segments, full detail --- */
    var du = 0.022;
    var first = Math.floor((G.u - 0.62) / du);
    var nearSteps = Math.round(1.30 / du);
    for (var i = 0; i < nearSteps; i++) {
      var a = (first + i) * du, b = a + du;
      var alt = (first + i) & 1;
      for (var c = 0; c < 3; c++) {
        var base;
        if (c === 1) base = alt ? ROAD_A : ROAD_B;
        else base = alt ? CURB_DIM : (c === 0 ? CURB_A : CURB_B);
        quad(surf(a, rows[c]), surf(a, rows[c + 1]),
             surf(b, rows[c + 1]), surf(b, rows[c]), base);
      }
      /* dashed edge + centre lines: the strongest speed cue a flat ribbon has */
      if (!alt) {
        quad(surf(a, W - 0.07), surf(a, W), surf(b, W), surf(b, W - 0.07),
             { r: 130, g: 240, b: 255 });
        quad(surf(a, -W), surf(a, -W + 0.07), surf(b, -W + 0.07), surf(b, -W),
             { r: 130, g: 240, b: 255 });
        quad(surf(a, -0.06), surf(a, 0.06), surf(b, 0.06), surf(b, -0.06),
             { r: 96, g: 112, b: 136 });
      }
    }

    /* --- far zone: the shape of the loop, nothing else --- */
    var fd = 0.08;
    var f0 = (first + nearSteps) * du;
    var fSteps = Math.round((4.4 - 1.30) / fd);
    for (var k = 0; k < fSteps; k++) {
      var fa = f0 + k * fd, fb = fa + fd;
      quad(surf(fa, -W), surf(fa, W), surf(fb, W), surf(fb, -W), FAR_ROAD);
      /* rim on both edges, every other segment: a Möbius strip is recognised
       * by its single continuous edge, so the edge is what gets drawn */
      if ((k & 1) === 0) {
        quad(surf(fa, -W), surf(fa, -W + 0.16), surf(fb, -W + 0.16), surf(fb, -W), RIM);
        quad(surf(fa, W - 0.16), surf(fa, W), surf(fb, W), surf(fb, W - 0.16), RIM);
      }
    }
  }

  /* Two lines are painted on the strip, and both rotate with the surface:
   * a checkered start/finish at u = 0 and a two-colour band at u = π, the
   * half-twist. Drive half a lap and you meet the twist band upside down
   * relative to the finish line — the clearest one-frame proof that the map is
   * not a ring with a stripe painted on it. */
  function paintedBand(u0, colA, colB) {
    var half = 0.030;
    var cols = 8;
    for (var r = 0; r < 2; r++) {
      var ua = u0 + (r ? 0 : -half), ub = u0 + (r ? half : 0);
      for (var c = 0; c < cols; c++) {
        var v0 = -W + (2 * W * c) / cols, v1 = -W + (2 * W * (c + 1)) / cols;
        quad(surf(ua, v0), surf(ua, v1), surf(ub, v1), surf(ub, v0),
             ((c + r) & 1) ? colA : colB);
      }
    }
  }
  function drawBands() {
    var twoPi = Math.PI * 2;
    /* nearest multiples of 0 and π in front of the kart */
    var k0 = Math.floor((G.u - 0.7) / twoPi);
    for (var j = 0; j < 2; j++) {
      var fin = (k0 + j) * twoPi;
      var duF = fin - G.u;
      if (duF > -0.6 && duF < 1.25) {
        paintedBand(fin, { r: 232, g: 238, b: 246 }, { r: 26, g: 32, b: 42 });
      }
      var tw = fin + Math.PI;
      var duT = tw - G.u;
      if (duT > -0.6 && duT < 1.25) {
        paintedBand(tw, { r: 255, g: 127, b: 208 }, { r: 86, g: 216, b: 255 });
      }
    }
  }

  /* A gate every quarter lap. Besides giving the player something to aim at, it
   * is the only cue that the surface is twisting: the frame a gate is built in
   * is the strip's own, so after half a lap the gate arrives upside down
   * relative to the one you just passed. */
  var GATES = [
    { r: 86, g: 216, b: 255 },
    { r: 255, g: 127, b: 208 },
    { r: 167, g: 139, b: 250 },
    { r: 245, g: 185, b: 92 }
  ];
  function drawGates() {
    var q = Math.PI * 0.5;
    var first = Math.ceil((G.u + 0.06) / q) * q;
    for (var i = 0; i < 2; i++) {
      var u = first + i * q;
      var du = u - G.u;
      if (du < 0.05 || du > 1.6) continue;
      var fr = frameAt(u, 0);
      var col = GATES[Math.round(u / q) % 4];
      boxAt(fr, -W + 0.3, 1.7, 0, 0.14, 1.7, 0.14, col);
      boxAt(fr, W - 0.3, 1.7, 0, 0.14, 1.7, 0.14, col);
      boxAt(fr, 0, 3.42, 0, W - 0.16, 0.1, 0.14, col);
    }
  }

  function drawObstacles() {
    for (var i = 0; i < G.obstacles.length; i++) {
      var o = G.obstacles[i];
      if (o.hit) continue;
      var du = o.u - G.u;
      if (du < -0.12 || du > 1.4) continue;
      var fr = frameAt(o.u, o.v);
      var col = TYPE_COLOR[o.type] || TYPE_COLOR.block;
      if (o.type === 'block') {
        boxAt(fr, 0, 0.62, 0, 0.62, 0.62, 0.62, col);
        boxAt(fr, 0, 1.32, 0, 0.17, 0.17, 0.17, { r: 255, g: 236, b: 190 });
      } else if (o.type === 'cone') {
        boxAt(fr, 0, 0.42, 0, 0.5, 0.42, 0.5, col);
        boxAt(fr, 0, 0.94, 0, 0.21, 0.21, 0.21, { r: 255, g: 246, b: 214 });
      } else if (o.type === 'duo') {
        boxAt(fr, -0.62, 0.42, 0, 0.36, 0.42, 0.36, col);
        boxAt(fr, 0.62, 0.42, 0, 0.36, 0.42, 0.36, col);
      } else {
        boxAt(fr, 0, 0.44, 0, 0.46, 0.44, 0.46, col);
        boxAt(fr, 0, 0.96, 0, 0.19, 0.19, 0.19, { r: 240, g: 250, b: 255 });
      }
    }
  }

  function drawKart() {
    var lean = clamp(-G.v * 0.16, -0.3, 0.3);
    var fr = frameAt(G.u, G.v);
    var c = Math.cos(lean), s = Math.sin(lean);
    var r = { x: fr.r.x * c + fr.n.x * s, y: fr.r.y * c + fr.n.y * s, z: fr.r.z * c + fr.n.z * s };
    var n = { x: fr.n.x * c - fr.r.x * s, y: fr.n.y * c - fr.r.y * s, z: fr.n.z * c - fr.r.z * s };
    var kf = { p: fr.p, r: r, n: n, f: fr.f };
    var body = G.invuln > 0 ? { r: 200, g: 210, b: 230 } : { r: 255, g: 96, b: 140 };
    var dark = { r: 34, g: 40, b: 52 };
    /* Suspension: with the camera locked to the kart, a perfectly rigid body
     * reads as a photo pasted on the road. Two centimetres of bob tied to the
     * distance travelled is enough to say "this thing is rolling". */
    var bob = Math.sin(G.dist * 2.4) * 0.022;

    var wheels = [[-0.55, 0.52], [0.55, 0.52], [-0.55, -0.5], [0.55, -0.5]];
    for (var w = 0; w < wheels.length; w++) {
      var wx = wheels[w][0], wz = wheels[w][1];
      boxAt(kf, wx, 0.26, wz, 0.2, 0.26, 0.32, dark);
      /* A bright stud orbiting the wheel face at the real rolling speed:
       * distance / radius. Without it the wheels are four static boxes. */
      var ph = G.dist / 0.26 + (w > 1 ? 0.7 : 0);
      boxAt(kf, wx + (wx < 0 ? -0.21 : 0.21), 0.26 + Math.sin(ph) * 0.15,
            wz + Math.cos(ph) * 0.15, 0.035, 0.055, 0.055,
            { r: 190, g: 210, b: 235 });
    }
    boxAt(kf, 0, 0.52 + bob, 0.02, 0.5, 0.22, 0.85, body);
    boxAt(kf, 0, 0.78 + bob, -0.06, 0.32, 0.16, 0.4, { r: 255, g: 214, b: 232 });
    boxAt(kf, 0, 0.86 + bob, 0.86, 0.4, 0.06, 0.1, { r: 220, g: 240, b: 255 });
    if (G.boost > 0) {
      octa(kf, 0, 0.5, -1.05, 0.22 + Math.random() * 0.08,
           { r: 120, g: 230, b: 255 }, 0.55);
    }
  }

  /* Speed streaks at the edges of the frame. Screen space on purpose: they are
   * about the camera, not about anything on the track, and 20 line segments
   * cost less than one obstacle. */
  function drawStreaks() {
    var v = Math.min(1.35, G.speed / MAX_SPEED);
    if (v < 0.55) return;
    var k = (v - 0.55) / 0.8;
    ctx.strokeStyle = 'rgba(160, 225, 255,' + (0.04 + 0.08 * k) + ')';
    ctx.lineWidth = 1;
    for (var i = 0; i < 12; i++) {
      var side = (i & 1) ? 1 : -1;
      var fx = 0.30 + ((i * 137) % 100) / 100 * 0.19;    /* outer fifth only */
      var x = Wpx * 0.5 + side * Wpx * fx;
      var y = ((i * 271 + G.dist * 300) % (Hpx + 160)) - 80;
      var len = 14 + 30 * k;
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(x + side * 4 * k, y + len);
      ctx.stroke();
    }
  }

  function paint() {
    setCamera();
    /* A little focal kick with speed: the same trick as a dolly zoom, cheap
     * and it makes 90 km/h feel different from 40. */
    var fovNow = FOV + 0.10 * Math.min(1, G.speed / MAX_SPEED);
    focal = (Hpx * 0.5) / Math.tan(fovNow * 0.5);
    tris.length = 0;
    nTri = 0;
    drawSky();
    drawTrack();
    drawBands();
    drawGates();
    drawObstacles();
    drawKart();
    tris.sort(function (a, b) { return b.z - a.z; });
    for (var i = 0; i < tris.length; i++) {
      var t = tris[i];
      ctx.globalAlpha = t.a;
      ctx.fillStyle = t.fill;
      ctx.beginPath();
      ctx.moveTo(t.x1, t.y1);
      ctx.lineTo(t.x2, t.y2);
      ctx.lineTo(t.x3, t.y3);
      ctx.closePath();
      ctx.fill();
    }
    ctx.globalAlpha = 1;
    drawStreaks();
    drawInset();
  }

  /* The map inset: the whole strip seen from outside the loop, with the kart's
   * position on it. A chase camera sits *inside* the twist — from there a
   * Möbius strip and a banked ring look the same — so the shape is shown from a
   * viewpoint the player can never occupy. Its own projection, deliberately:
   * the main camera is a per-frame function of the kart and must not be
   * disturbed by a 140-pixel diagram. */
  function drawInset() {
    var size = Math.round(Math.max(96, Math.min(150, Math.min(Wpx, Hpx) * 0.24)));
    var pad = 12, x0 = Wpx - size - pad, y0 = pad;
    /* On a phone the HUD wraps to two rows, so the inset drops below it
     * instead of sitting under the top-right corner. */
    if (Wpx < 620) { size = 88; x0 = Wpx - size - 8; y0 = 84; }

    /* A shallow angle: from straight above, a Möbius strip is a disc with a
     * shading difference. Seen at ~19 degrees the band twists through the view
     * and the half-twist is unmistakable. */
    var eye = { x: R * 0.30, y: R * 0.58, z: R * 1.52 };
    var f = { x: 0, y: 0, z: 0 };
    f.x = -eye.x; f.y = -eye.y; f.z = -eye.z;
    var fl0 = Math.hypot(f.x, f.y, f.z) || 1;
    f.x /= fl0; f.y /= fl0; f.z /= fl0;
    var r = { x: f.z, y: 0, z: -f.x };          /* right = f x up */
    var rl = Math.hypot(r.x, r.y, r.z) || 1;
    r.x /= rl; r.y /= rl; r.z /= rl;
    var u2 = {
      x: r.y * f.z - r.z * f.y,
      y: r.z * f.x - r.x * f.z,
      z: r.x * f.y - r.y * f.x
    };
    var focalI = size * 0.56;
    function proj(p) {
      var dx = p.x - eye.x, dy = p.y - eye.y, dz = p.z - eye.z;
      var cz = dx * f.x + dy * f.y + dz * f.z;
      if (cz < 8) return null;
      return {
        x: x0 + size * 0.5 + focalI * (dx * r.x + dy * r.y + dz * r.z) / cz,
        y: y0 + size * 0.5 - focalI * (dx * u2.x + dy * u2.y + dz * u2.z) / cz,
        z: cz
      };
    }

    ctx.save();
    ctx.beginPath();
    ctx.rect(x0, y0, size, size);
    ctx.clip();
    var bg = ctx.createLinearGradient(x0, y0, x0, y0 + size);
    bg.addColorStop(0, 'rgba(8,13,20,0.72)');
    bg.addColorStop(1, 'rgba(8,13,20,0.46)');
    ctx.fillStyle = bg;
    ctx.fillRect(x0, y0, size, size);

    var steps = 72, du = TWO_PI / steps;
    var iw = W * 2.4;    /* diagram width: at true scale the ribbon is a hairline */
    var faces = [];
    for (var i = 0; i < steps; i++) {
      var a = i * du, b = a + du;
      var p1 = proj(surf(a, -iw)), p2 = proj(surf(a, iw));
      var p3 = proj(surf(b, iw)), p4 = proj(surf(b, -iw));
      if (!p1 || !p2 || !p3 || !p4) continue;
      var n = normalAt(a);
      /* Colour by which way the face points, not by lighting: on a Möbius strip
       * the surface you are looking at flips *side* halfway round — that flip
       * is the whole point, and a uniform shade would hide it. Teal = the face
       * points up here, violet = it points down. */
      var lit = 0.55 + 0.45 * Math.abs(n.y);
      var near = wrapDu(a - G.u);
      var hot = Math.abs(near) < 0.14;
      var base = n.y >= 0 ? [86, 216, 255] : [178, 140, 255];
      faces.push({
        z: (p1.z + p3.z) * 0.5,
        fill: hot ? 'rgba(255,120,176,0.98)'
                  : 'rgba(' + Math.round(base[0] * lit) + ',' +
                    Math.round(base[1] * lit) + ',' +
                    Math.round(base[2] * lit) + ',0.9)',
        p: [p1, p2, p3, p4]
      });
    }
    faces.sort(function (q, w) { return w.z - q.z; });
    for (var k = 0; k < faces.length; k++) {
      var fc = faces[k];
      ctx.fillStyle = fc.fill;
      ctx.beginPath();
      ctx.moveTo(fc.p[0].x, fc.p[0].y);
      ctx.lineTo(fc.p[1].x, fc.p[1].y);
      ctx.lineTo(fc.p[2].x, fc.p[2].y);
      ctx.lineTo(fc.p[3].x, fc.p[3].y);
      ctx.closePath();
      ctx.fill();
    }

    /* the half-twist landmark and the kart */
    for (var mi = 0; mi < 2; mi++) {
      var uT = Math.round(G.u / (Math.PI * 2)) * Math.PI * 2 + Math.PI * mi;
      var q1 = proj(surf(uT, -iw)), q2 = proj(surf(uT, iw));
      if (q1 && q2) {
        ctx.strokeStyle = mi === 0 ? 'rgba(232,238,246,0.9)' : 'rgba(255,214,92,0.9)';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(q1.x, q1.y);
        ctx.lineTo(q2.x, q2.y);
        ctx.stroke();
      }
    }
    var kp = proj(surf(G.u, G.v));
    if (kp) {
      ctx.fillStyle = '#ff5f9e';
      ctx.beginPath();
      ctx.arc(kp.x, kp.y, 3.2, 0, TWO_PI);
      ctx.fill();
      ctx.strokeStyle = 'rgba(255,255,255,0.75)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(kp.x, kp.y, 5.4, 0, TWO_PI);
      ctx.stroke();
    }
    ctx.restore();

    ctx.strokeStyle = 'rgba(255,255,255,0.12)';
    ctx.lineWidth = 1;
    ctx.strokeRect(x0 + 0.5, y0 + 0.5, size - 1, size - 1);
    ctx.fillStyle = 'rgba(150,170,190,0.85)';
    ctx.font = '9px ui-monospace, Menlo, monospace';
    ctx.fillText('MÖBIUS', x0 + 6, y0 + size - 6);
  }

  /* ------------------------------------------------------------------ */
  /* the model: the only source of obstacles                             */
  /* ------------------------------------------------------------------ */
  /* Same origin, always. The resident model comes from /v1/models, so a
     switch on the server is picked up by the next obstacle decision. */
  function apiBase() { return location.origin + '/v1'; }
  function apiHeaders(json) {
    var h = {};
    if (json) h['Content-Type'] = 'application/json';
    return h;
  }

  /* Short on purpose: this prompt is prefilled on every obstacle, Chinese costs
   * three bytes a character in this tokenizer, and a demo checkpoint with a
   * 512-token window refuses a long rubric outright. Only `why` — the part a
   * player reads — may be Chinese. */
  /* The obstacle decision is discrete on purpose: four kinds times three lane
   * positions, one letter each. A continuous lane offset and gap would have to
   * be generated token by token and then repaired when the model writes prose
   * around them; as options they come out of one prefill. Distance is derived
   * locally from the kart's speed, which is also what keeps the road readable:
   * the player sees where the thing will be, not just what kind it is. */
  var LANES = [
    { key: 'left', at: -0.55, name: 'left lane' },
    { key: 'center', at: 0, name: 'centre lane' },
    { key: 'right', at: 0.55, name: 'right lane' }
  ];
  var DECISIONS = [];
  for (var dti = 0; dti < TYPES.length; dti++)
    for (var dli = 0; dli < LANES.length; dli++)
      DECISIONS.push({
        key: TYPES[dti] + '-' + LANES[dli].key,
        type: TYPES[dti],
        lane: LANES[dli].at,
        label: TYPE_EN[TYPES[dti]] + ' in the ' + LANES[dli].name
      });

  /* Real-time random decisions. A /v1/systemone readout is deterministic for
     a fixed prompt, and a fixed A..L order would put the model's letter bias
     on the same obstacle every time. Three things vary per request, none of
     which places an obstacle locally: a Fisher-Yates shuffle of the option
     letters, a variation seed in the state, and one randomly drawn pressure
     point. The model still picks the kind and lane out of one prefill; the
     seed and the resulting distribution are shown in the sidebar and stored
     in the sqlite event, so the randomness is auditable rather than hidden. */
  var FOCUS = [
    'force a late lane change at this speed',
    'take away the line the kart is holding now',
    'chain with the obstacles already ahead',
    'punish the centre line on the flipped side',
    'leave only a narrow gap near the rim',
    'make the next obstacle hard to read at this speed'
  ];
  function shuffled(src) {
    var a = src.slice();
    for (var i = a.length - 1; i > 0; i--) {
      var j = Math.floor(Math.random() * (i + 1));
      var t = a[i]; a[i] = a[j]; a[j] = t;
    }
    return a;
  }
  /* Normalised entropy of the option distribution: 0 = the model is certain,
     1 = it spread its answer mass evenly. Shown as "决策分散度" so a player
     can see whether the variation actually changed the decision. */
  function entropyOf(p) {
    var keys = Object.keys(p || {}), h = 0;
    for (var i = 0; i < keys.length; i++) {
      var v = Number(p[keys[i]]) || 0;
      if (v > 0) h -= v * Math.log(v);
    }
    return keys.length > 1 ? h / Math.log(keys.length) : 0;
  }
  var HARD_QUESTION =
    'Which obstacle makes this lap hardest for the player? Maximise difficulty: ' +
    'use the speed, the obstacles already ahead and the pressure point to pick ' +
    'the kind and lane that most constrain the racing line. Do not leave an easy ' +
    'way through. Choose one option.';

  var MODEL_ID = '', MODEL_PROMISE = null;
  function resolveModel() {
    if (MODEL_PROMISE) return;
    MODEL_PROMISE = fetch(apiBase() + '/models', { headers: apiHeaders(false), cache: 'no-store' })
      .then(function (r) { if (authRedirect(r)) return null; return r.ok ? r.json() : null; })
      .then(function (j) {
        if (!j || !Array.isArray(j.data)) return '';
        var pick = null;
        for (var i = 0; i < j.data.length; i++) {
          if (j.data[i].cwick && j.data[i].cwick.loaded) { pick = j.data[i]; break; }
        }
        MODEL_ID = (pick && pick.id) || (j.data[0] && j.data[0].id) || '';
        /* An empty answer must not be cached: a resolution that found nothing
         * would otherwise be handed to every later call, and one cold start —
         * the first spawn firing before the server can list its models —
         * would leave the game without obstacles for the whole run. */
        if (!MODEL_ID) MODEL_PROMISE = null;
        return MODEL_ID;
      })
      .catch(function () { MODEL_PROMISE = null; return ''; });
  }
  function ensureModel() {
    if (MODEL_ID) return Promise.resolve(MODEL_ID);
    resolveModel();
    return MODEL_PROMISE || Promise.resolve('');
  }
  function forgetModel() { MODEL_ID = ''; MODEL_PROMISE = null; }

  async function askAgent() {
    var retried = false;
    for (;;) {
      var model = await ensureModel();
      if (!model) throw new Error('no model published by /v1/models');
      var seed = Math.random().toFixed(4);
      var focus = FOCUS[Math.floor(Math.random() * FOCUS.length)];
      /* The strip really turns over every lap; reporting which side is up is
         a genuine, changing state fact rather than invented noise. */
      var side = Math.cos(G.u * 0.5) >= 0 ? 'outer' : 'inner (flipped)';
      var state =
        'Bubble-kart race on a Mobius loop. Lap ' + G.lap +
        ', speed ' + Math.round(G.speed * KMH) + ' km/h' +
        ', laps completed ' + (G.lap - 1) +
        ', hits ' + G.hits +
        ', obstacles ahead ' + aheadCount() +
        ', spawned so far ' + G.spawned +
        ', difficulty level ' + aiLevel() + '/5 (raise it)' +
        ', surface side ' + side +
        ', variation seed ' + seed + '; ' +
        'pressure point: ' + focus + '.';
      /* criteria is an ordered JSON object on the server: member order becomes
         the A..L letters. Shuffling it is what makes the same kart state ask a
         genuinely different question on the next frame. */
      var order = shuffled(DECISIONS);
      var crit = {};
      for (var di = 0; di < order.length; di++) crit[order[di].key] = order[di].label;
      var t0 = Date.now();
      var res = await fetch(apiBase() + '/systemone', {
        method: 'POST',
        headers: apiHeaders(true),
        body: JSON.stringify({
          model: model,
          state: state,
          questions: {
            obstacle: {
              type: 'choice',
              instructions: HARD_QUESTION,
              criteria: crit
            }
          }
        })
      });
      if (authRedirect(res)) throw new Error('authentication required');
      if (!res.ok) {
        var text = '';
        try { text = await res.text(); } catch (e) {}
        /* A stale name in the settings tab would otherwise refuse every call
         * forever: forget the name, discover the resident model, retry once. */
        if (!retried && res.status === 400 && /not served by this endpoint/i.test(text)) {
          retried = true;
          forgetModel(model);
          record('agent', '设置里的模型 <b>' + esc(model) +
                 '</b> 不是当前服务提供的那个，改用常驻模型重试', {});
          continue;
        }
        throw new Error('HTTP ' + res.status + ' ' + text.slice(0, 140));
      }
      var j = await res.json();
      var a = (j.answers && j.answers.obstacle) || {};
      var probs = a.probabilities || {};
      return { plan: parseDecision(a), raw: JSON.stringify(probs),
               probs: probs, ms: Date.now() - t0,
               seed: seed, focus: focus, entropy: entropyOf(probs) };
    }
  }

  function parseDecision(a) {
    if (!a || !a.choice) return null;
    var key = String(a.choice);
    var d = null;
    for (var i = 0; i < DECISIONS.length; i++)
      if (DECISIONS[i].key === key) { d = DECISIONS[i]; break; }
    if (!d) return null;
    var p = (a.probabilities && a.probabilities[key]) || 0;
    return {
      type: d.type,
      lane: d.lane,
      /* Faster kart, more room to react: the gap grows with speed instead of
       * being one more thing for the model to get wrong. */
      gap: Math.round(clamp(45 + G.speed * KMH / 6, 45, 90)),
      why: TYPE_NAME[d.type] + ' ' + (p * 100).toFixed(0) + '%',
      confidence: Number(a.confidence) || 0
    };
  }

  /* One request in flight, two decisions buffered: a kart covers ~13 m in the
   * second a large model may take to answer, so the queue — not the frame
   * loop — is what keeps the road populated. With no local generator behind
   * this, an empty queue means an empty road; the HUD says so. */
  function pumpAgent(dt) {
    if (G.mode !== 'running') return;
    if (G.asking || G.plan.length >= 2) return;
    G.askAgainIn -= dt;
    if (G.askAgainIn > 0) return;
    G.asking = true;
    G.aiState = 'busy';
    askAgent().then(function (r) {
      G.asking = false;
      AGENT.calls++;
      if (!r.plan) {
        AGENT.fail++;
        G.askFail++;
        G.askAgainIn = 2 + G.askFail * 1.5;
        G.aiState = 'waiting';
        record('agent', '模型回答无法解析，等它下一次决定', { raw: r.raw.slice(0, 120) });
        return;
      }
      G.askFail = 0;
      G.aiState = G.plan.length >= 2 ? 'idle' : 'busy';
      AGENT.ok++;
      AGENT.msTotal += r.ms;
      AGENT.msN++;
      r.plan.ms = r.ms;
      r.plan.raw = r.raw.slice(0, 240);
      r.plan.seed = r.seed;
      r.plan.focus = r.focus;
      r.plan.entropy = r.entropy;
      G.plan.push(r.plan);
      AGENT.last = r.plan;
      AGENT.lastSeed = r.seed;
      AGENT.lastFocus = r.focus;
      AGENT.lastEntropy = r.entropy;
      record('agent', '模型决定：<b>' + TYPE_NAME[r.plan.type] + '</b> · 横向 ' +
             fmt(r.plan.lane, 2) + ' · 前方 ' + r.plan.gap + ' m · ' + r.ms + ' ms' +
             ' · 种子 ' + esc(r.seed) + ' · 分散度 ' + Math.round(r.entropy * 100) + '%' +
             (r.plan.why ? ' · ' + esc(r.plan.why) : ''),
             { type: r.plan.type, lane: r.plan.lane, gap: r.plan.gap,
               ms: r.ms, why: r.plan.why, raw: r.plan.raw,
               seed: r.seed, focus: r.focus, entropy: r.entropy,
               confidence: r.plan.confidence });
    }).catch(function (err) {
      G.asking = false;
      AGENT.calls++;
      AGENT.fail++;
      G.askFail++;
      G.askAgainIn = 2 + G.askFail * 1.5;
      G.aiState = 'down';
      record('agent', '模型不可用（' +
             esc(String(err && err.message ? err.message : err).slice(0, 90)) +
             '）· 赛道暂时不放路障', {});
    });
  }

  /* ------------------------------------------------------------------ */
  /* game                                                                */
  /* ------------------------------------------------------------------ */
  function aiLevel() { return Math.min(5, G.lap); }
  function diffName(l) { return DIFF_NAME[l - 1] || '极致'; }
  function aheadCount() {
    var n = 0;
    for (var i = 0; i < G.obstacles.length; i++) {
      if (!G.obstacles[i].hit && wrapDu(G.obstacles[i].u - G.u) > 0) n++;
    }
    return n;
  }
  function obstacleHalf(type) {
    return type === 'block' ? 0.62 : type === 'duo' ? 1.0 : 0.5;
  }

  /* Called when the kart has passed the point the last decision was placed at,
   * and only with a decision in hand. */
  function placeObstacle() {
    var plan = G.plan.shift();
    if (!plan) return false;
    var lane = plan.lane * LIM;
    G.obstacles.push({
      u: G.u + plan.gap / R, v: lane, type: plan.type, hit: false,
      hw: obstacleHalf(plan.type), ms: plan.ms
    });
    G.spawned++;
    G.byType[plan.type]++;
    /* The model decides *what* and *which lane*; spacing is derived locally
       from speed (see parseDecision). Difficulty now tightens that spacing as
       laps rise, but never below the reaction window: a "hardest possible"
       obstacle the player cannot see is just an unfair collision. */
    var breathe = Math.max(10, 26 - (aiLevel() - 1) * 4);
    G.nextPlaceU = G.u + (plan.gap + breathe) / R;
    record('spawn', '模型生成 <b>' + TYPE_NAME[plan.type] + '</b> · 难度 ' +
           diffName(aiLevel()) + ' · ' + (plan.ms || 0) + ' ms',
           { type: plan.type, lane: Math.round(lane * 100) / 100, gap: plan.gap,
             level: aiLevel(), ms: plan.ms || 0,
             why: plan.why || '', raw: plan.raw || '',
             seed: plan.seed || '', focus: plan.focus || '',
             entropy: plan.entropy || 0 });
    return true;
  }

  function startRun() {
    G.mode = 'running';
    G.u = 0.6; G.v = 0; G.speed = 0; G.lap = 1;
    G.dist = 0; G.score = 0; G.hits = 0; G.spawned = 0;
    G.byType = { block: 0, cone: 0, duo: 0, roll: 0 };
    G.obstacles = []; G.plan = []; G.asking = false;
    G.askFail = 0; G.askAgainIn = 0; G.boost = 0; G.boostCd = 0; G.invuln = 0;
    G.elapsed = 0; G.events = []; G.evSeq = 0; evBuf.length = 0;
    G.lapStart = 0; G.bestLap = 0; G.runStart = Date.now();
    G.nextPlaceU = G.u;
    G.camU = G.u; G.camV = 0;
    G.aiState = 'busy';
    AGENT.calls = 0; AGENT.ok = 0; AGENT.fail = 0;
    AGENT.msTotal = 0; AGENT.msN = 0; AGENT.last = null;
    AGENT.lastSeed = ''; AGENT.lastFocus = ''; AGENT.lastEntropy = 0;
    G.gameId = 'k' + Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
    G.startedAt = Math.floor(Date.now() / 1000);
    $('overlay').classList.add('off');
    setStatus('比赛中 · 模型实时随机决策，目标是把跑圈难度拉满');
    putGame({ started_at: G.startedAt, ended_at: 0, outcome: '', moves: 0 });
    record('lap', '比赛开始 · 第 1 圈', {});
    paintSide(true);
  }

  function endRun() {
    if (G.mode === 'idle') return;
    G.mode = 'idle';
    var laps = G.lap - 1;
    BEST.score = Math.max(BEST.score, Math.round(G.score));
    BEST.lap = Math.max(BEST.lap, laps + 1);
    if (G.bestLap && (!BEST.lapMs || G.bestLap < BEST.lapMs)) BEST.lapMs = G.bestLap;
    LIFE.runs++; LIFE.laps += laps; LIFE.dist += G.dist;
    LIFE.hits += G.hits; LIFE.spawned += G.spawned;
    lsWrite({ best: BEST, lifetime: LIFE });
    putState('best', BEST);
    putState('lifetime', LIFE);
    putGame({ started_at: G.startedAt, ended_at: Math.floor(Date.now() / 1000),
              outcome: 'stop', moves: G.spawned });
    record('end', '本局结束 · ' + laps + ' 圈 · ' + fmt(G.dist, 0) + ' m · 撞 ' +
           G.hits + ' 次 · 模型放 ' + G.spawned + ' 个路障',
           { score: Math.round(G.score), lap_ms: G.bestLap || 0 });
    showSummary(laps);
    setStatus('已结束 · ' + laps + ' 圈');
    paintSide(true);
    setTimeout(loadHistory, 700);
  }

  function showSummary(laps) {
    var p = document.querySelector('#overlay .panel');
    if (!p) return;
    p.innerHTML =
      '<h2>这一局跑了 ' + laps + ' 圈</h2>' +
      '<p class="sub">里程 <b>' + fmt(G.dist, 0) + ' m</b> · 最快圈 <b>' +
      secs(G.bestLap) + '</b> · 撞击 <b>' + G.hits + ' 次</b></p>' +
      '<p class="sub">模型放了 <b>' + G.spawned + '</b> 个路障（全部来自 AI），' +
      '调用 <b>' + AGENT.calls + '</b> 次，平均 <b>' +
      (AGENT.msN ? Math.round(AGENT.msTotal / AGENT.msN) : '—') + ' ms</b></p>' +
      '<button id="btnStart">再跑一局</button>' +
      '<p class="keys">← → 转向 · 空格加速 · 手机上拖动转向</p>';
    var b = $('btnStart');
    if (b) b.addEventListener('click', startRun);
    $('overlay').classList.remove('off');
  }

  function step(dt) {
    if (G.mode !== 'running') return;
    G.elapsed += dt;
    if (G.invuln > 0) G.invuln -= dt;

    var top = MAX_SPEED + Math.min(9, (G.lap - 1) * 2.2);
    if (G.boostCd > 0) G.boostCd -= dt;
    if (G.boost > 0) G.boost -= dt;
    var target = G.boost > 0 ? top * 1.5 : top;
    G.speed += (target - G.speed) * Math.min(1, dt * 1.5);
    if (G.speed < 0) G.speed = 0;

    var steer = (keys.left ? -1 : 0) + (keys.right ? 1 : 0) + dragSteer;
    G.v = clamp(G.v + steer * 8.0 * dt, -LIM, LIM);

    G.u += (G.speed * dt) / R;
    G.dist += G.speed * dt;

    /* The camera trails the kart instead of being welded to it. Nothing moves
     * on screen when the eye is an exact function of the kart's position, so a
     * 90 km/h run looked like a still photo; a spring a few metres behind makes
     * acceleration and steering read as motion. */
    var lag = Math.min(1, dt * 9);
    G.camU += (G.u - G.camU) * lag;
    G.camV += (G.v - G.camV) * lag;
    G.score += G.speed * dt * 0.35;

    var lapNow = Math.floor(G.u / TWO_PI) + 1;
    if (lapNow > G.lap) {
      G.lap = lapNow;
      var lapMs = Math.round((Date.now() - G.runStart) - G.lapStart);
      G.lapStart += lapMs;
      if (!G.bestLap || lapMs < G.bestLap) G.bestLap = lapMs;
      G.score += 150;
      record('lap', '第 <b>' + (G.lap - 1) + '</b> 圈完成 · ' + secs(lapMs) +
             (G.bestLap === lapMs ? ' · 最快' : ''), { lap_ms: lapMs });
    }

    pumpAgent(dt);
    /* Only the model places obstacles: this consumes one decision per stretch
     * of road and does nothing at all when the queue is empty. */
    if (G.u > G.nextPlaceU && G.plan.length && G.obstacles.length < 8) placeObstacle();

    for (var i = G.obstacles.length - 1; i >= 0; i--) {
      var o = G.obstacles[i];
      var d = wrapDu(o.u - G.u);
      if (d < -0.35) { G.obstacles.splice(i, 1); continue; }
      if (o.hit) continue;
      if (G.invuln <= 0 && Math.abs(d) < 0.035 && Math.abs(o.v - G.v) < o.hw + 0.55) {
        o.hit = true;
        G.hits++;
        G.invuln = 1.1;
        G.speed *= 0.35;
        G.score = Math.max(0, G.score - 30);
        record('hit', '撞上 <b>' + TYPE_NAME[o.type] + '</b> · 速度掉到 ' +
               Math.round(G.speed * KMH) + ' km/h', { type: o.type });
      }
    }
  }

  /* ------------------------------------------------------------------ */
  /* UI                                                                  */
  /* ------------------------------------------------------------------ */
  var statusEl = $('status'), sideStatus = $('sideStatus'), liveEl = $('live');
  function setStatus(s) {
    if (statusEl) statusEl.textContent = s;
    if (sideStatus) sideStatus.textContent = s;
  }
  function setLive(kind) { if (liveEl) liveEl.className = 'dot' + (kind ? ' ' + kind : ''); }

  function cell(label, value, unit, cls) {
    return '<div class="cell' + (cls ? ' ' + cls : '') + '"><span>' + label + '</span><b>' +
      value + (unit ? '<i>' + unit + '</i>' : '') + '</b></div>';
  }

  function aiLabel() {
    if (G.mode !== 'running') return G.aiState === 'down' ? '模型离线' : '待命';
    if (G.aiState === 'busy') return '模型思考中…';
    if (G.aiState === 'waiting') return '等待模型';
    if (G.aiState === 'down') return '模型离线 · 暂不放路障';
    return '就绪';
  }

  function paintHud() {
    if (!$('hSpeed')) return;
    $('hSpeed').textContent = Math.round(G.speed * KMH);
    $('hLap').textContent = Math.max(0, G.lap - 1);
    $('hScore').textContent = Math.round(G.score);
    $('hMode').textContent = 'AI 待放 ' + G.plan.length + ' 个 · ' + aiLabel();
  }

  var sideAt = 0;
  function paintSide(force) {
    var now = performance.now();
    if (!force && now - sideAt < 250) return;
    sideAt = now;
    if (!$('aiStats')) return;
    $('aiStats').innerHTML =
      cell('已放路障', G.spawned, '个', 'accent') +
      cell('模型调用', AGENT.calls, '次') +
      cell('失败', AGENT.fail, '次', AGENT.fail ? 'pink' : '') +
      cell('平均响应', AGENT.msN ? Math.round(AGENT.msTotal / AGENT.msN) : '—', 'ms') +
      cell('待放队列', G.plan.length, '个') +
      cell('当前难度', diffName(aiLevel()), '') +
      cell('随机种子', AGENT.lastSeed || '—', '') +
      cell('决策分散度', AGENT.lastSeed ? Math.round(AGENT.lastEntropy * 100) : '—', '%');

    var max = Math.max(1, G.byType.block, G.byType.cone, G.byType.duo, G.byType.roll);
    $('typeBars').innerHTML = TYPES.map(function (t) {
      var c = TYPE_COLOR[t];
      return '<div class="bar-row"><span>' + TYPE_NAME[t].slice(0, 2) + '</span>' +
        '<span class="bar"><i style="width:' +
        Math.round((G.byType[t] / max) * 100) + '%;background:rgb(' +
        c.r + ',' + c.g + ',' + c.b + ')"></i></span><b>' + G.byType[t] + '</b></div>';
    }).join('');

    var last = AGENT.last;
    $('aiLast').innerHTML = last
      ? '<b>' + TYPE_NAME[last.type] + '</b> · 横向 ' + fmt(last.lane, 2) +
        ' · 前方 ' + last.gap + ' m · ' + last.ms + ' ms' +
        (last.why ? '<br>“' + esc(last.why) + '”' : '') +
        (AGENT.lastSeed ? '<br><span class="mono">种子 ' + esc(AGENT.lastSeed) +
          ' · 分散度 ' + Math.round(AGENT.lastEntropy * 100) + '%</span>' +
          '<br><span class="muted">加压点：' + esc(AGENT.lastFocus) + '</span>' : '')
      : '—';

    $('carStats').innerHTML =
      cell('速度', Math.round(G.speed * KMH), 'km/h', 'accent') +
      cell('里程', fmt(G.dist, 0), 'm') +
      cell('圈数', Math.max(0, G.lap - 1), '/ ' + (BEST.lap - 1) + ' 最佳') +
      cell('最快圈', secs(G.bestLap), '') +
      cell('碰撞', G.hits, '次', G.hits ? 'pink' : '') +
      cell('本局时长', fmt(G.elapsed, 0), 's');
    $('runLine').textContent = '得分 ' + Math.round(G.score) + ' · 用时 ' +
      fmt(G.elapsed, 1) + ' s · 路障 ' + G.spawned + ' 个 · 平均每 ' +
      (G.spawned ? fmt(G.dist / G.spawned, 0) : '—') + ' m 一个';
    $('lifeLine').textContent = '对局 ' + LIFE.runs + ' · 总圈数 ' + LIFE.laps +
      ' · 总里程 ' + fmt(LIFE.dist / 1000, 1) + ' km · 碰撞 ' + LIFE.hits;
    if ($('bestScore')) $('bestScore').textContent = BEST.score;
    if ($('bestLap')) $('bestLap').textContent = Math.max(0, BEST.lap - 1);
    setLive(G.mode === 'running' ? (G.aiState === 'busy' ? 'busy' : 'on') : '');
  }

  function renderFeeds() {
    var feed = $('feed'), ev = $('eventFeed');
    var recent = G.events.slice(-14).reverse();
    if (feed) feed.innerHTML = recent.map(feedItem).join('') || '<li>还没有决策。</li>';
    if (ev) ev.innerHTML = G.events.slice(-16).reverse().map(feedItem).join('') || '<li>—</li>';
  }
  function feedItem(e) {
    var cls = e.kind === 'spawn' ? 'agent'
      : e.kind === 'hit' ? 'hit'
      : e.kind === 'lap' || e.kind === 'end' ? 'lap'
      : e.kind === 'agent' ? 'agent' : '';
    return '<li class="' + cls + '">' + fmt(e.t, 1) + 's · ' + e.text + '</li>';
  }

  /* ------------------------------------------------------------------ */
  /* history                                                             */
  /* ------------------------------------------------------------------ */
  var histGames = [], histSel = '';
  function loadHistory() {
    if (G.storeOk === false) { renderGames(lsGet('games') || []); return; }
    storeFetch('/v1/apps/kart/games?limit=20').then(function (r) {
      if (r.status === 503) { G.storeOk = false; renderGames(lsGet('games') || []); return null; }
      return r.ok ? r.json() : null;
    }).then(function (j) {
      if (!j) return;
      histGames = j.games || [];
      lsWrite({ games: histGames });
      renderGames(histGames);
    }).catch(function () { renderGames(lsGet('games') || []); });
  }
  function renderGames(list) {
    var ul = $('games');
    if (!ul) return;
    histGames = list || [];
    $('histEmpty').style.display = histGames.length ? 'none' : 'block';
    ul.innerHTML = histGames.map(function (g) {
      var when = new Date((g.ended_at || g.started_at) * 1000);
      return '<li data-id="' + g.game_id + '"' + (g.game_id === histSel ? ' class="on"' : '') + '>' +
        '<b>' + (g.moves || 0) + ' 个路障</b>' +
        '<span>' + when.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' }) +
        '</span></li>';
    }).join('');
    Array.prototype.forEach.call(ul.children, function (li) {
      li.addEventListener('click', function () { loadReplay(li.getAttribute('data-id')); });
    });
  }
  function loadReplay(id) {
    histSel = id;
    renderGames(histGames);
    storeFetch('/v1/apps/kart/events?game_id=' + encodeURIComponent(id)).then(function (r) {
      return r.ok ? r.json() : null;
    }).then(function (j) {
      var ul = $('replay');
      if (!ul) return;
      var evs = (j && j.events) || [];
      ul.innerHTML = evs.length ? evs.map(function (e) {
        var p = e.payload || {};
        var cls = e.kind === 'spawn' ? 'agent'
          : e.kind === 'hit' ? 'hit' : e.kind === 'lap' || e.kind === 'end' ? 'lap' : '';
        return '<li class="' + cls + '">' + fmt(p.t || 0, 1) + 's · ' +
          esc(p.text || e.kind) + '</li>';
      }).join('') : '<li>这一局没有记录。</li>';
    }).catch(function () {});
  }

  /* ------------------------------------------------------------------ */
  /* controls                                                            */
  /* ------------------------------------------------------------------ */
  /* ------------------------------------------------------------------ */
  /* share / QR                                                         */
  /* ------------------------------------------------------------------ */
  /* GET /v1/share?app=kart: the server answers with the LAN address a phone
   * can reach (or with shareable:false and a reason when the listener is
   * loopback-only), and GET /v1/share/qr draws that URL as a PNG. Both are
   * answered before the generation slot, so asking costs the model nothing. */
  var APP_ID = 'kart';
  var APP_ENTRY = '/apps/kart/index.html';

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
      var res = await fetch('/v1/share?app=' + APP_ID, { cache: 'no-store' });
      if (authRedirect(res)) return;
      if (!res.ok) throw new Error('HTTP ' + res.status);
      var j = await res.json();
      var url = j.app_url ||
        ((j.lan_urls && j.lan_urls.length) ? j.lan_urls[0] + APP_ENTRY : '');
      if (!j.shareable || !url) {
        urlEl.textContent = location.origin + APP_ENTRY;
        note.className = 'share-note bad';
        note.textContent = (j.warning || '当前服务没有可分享的局域网地址。') +
          ' 在这台 Mac 上用 ./start.sh（已带 --lan）或 cwick serve --lan 启动后重试。';
        return;
      }
      urlEl.textContent = url;
      img.src = '/v1/share/qr?text=' + encodeURIComponent(url);
      img.hidden = false;
      note.textContent = '手机与这台 Mac 连同一个 Wi-Fi，扫码后用浏览器打开即可独立运行；首次打开需要输入这台 Mac 的登录密码。';
    } catch (e) {
      urlEl.textContent = location.origin + APP_ENTRY;
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

  function bindUI() {
    if ($('share')) $('share').addEventListener('click', share);
    if ($('shareClose')) $('shareClose').addEventListener('click', closeShare);
    if ($('copyShare')) $('copyShare').addEventListener('click', copyShare);
    $('btnStart').addEventListener('click', startRun);
    $('btnStop').addEventListener('click', endRun);
    $('btnPause').addEventListener('click', function () {
      if (G.mode === 'running') { G.mode = 'paused'; setStatus('已暂停'); showPaused(); }
      else if (G.mode === 'paused') {
        G.mode = 'running';
        $('overlay').classList.add('off');
        setStatus('比赛中');
      }
    });
    $('quality').addEventListener('change', function (e) {
      QUALITY = Number(e.target.value) || 1;
      putState('quality', QUALITY);
      resize();
    });
    $('btnClear').addEventListener('click', function () {
      histSel = '';
      lsWrite({ games: [] });
      if (G.storeOk !== false) {
        storeFetch('/v1/apps/kart/games', { method: 'DELETE' }).catch(function () {});
      }
      renderGames([]);
      $('replay').innerHTML = '';
      BEST = { score: 0, lap: 1, lapMs: 0 };
      LIFE = { runs: 0, laps: 0, dist: 0, hits: 0, spawned: 0 };
      putState('best', BEST);
      putState('lifetime', LIFE);
      paintSide(true);
    });

    Array.prototype.forEach.call(document.querySelectorAll('.tab'), function (t) {
      t.addEventListener('click', function () {
        Array.prototype.forEach.call(document.querySelectorAll('.tab'), function (x) {
          x.classList.toggle('on', x === t);
        });
        var name = t.getAttribute('data-tab');
        Array.prototype.forEach.call(document.querySelectorAll('.pane'), function (p) {
          p.classList.toggle('on', p.id === 'tab-' + name);
        });
        if (name === 'hist') loadHistory();
      });
    });

    window.addEventListener('keydown', function (e) {
      if (e.key === 'ArrowLeft' || e.key === 'a' || e.key === 'A') keys.left = true;
      else if (e.key === 'ArrowRight' || e.key === 'd' || e.key === 'D') keys.right = true;
      else if (e.code === 'Space') { boost(); e.preventDefault(); }
      else if (e.key === 'p' || e.key === 'P') $('btnPause').click();
    });
    window.addEventListener('keyup', function (e) {
      if (e.key === 'ArrowLeft' || e.key === 'a' || e.key === 'A') keys.left = false;
      else if (e.key === 'ArrowRight' || e.key === 'd' || e.key === 'D') keys.right = false;
    });

    bindHold($('tLeft'), function (on) { keys.left = on; });
    bindHold($('tRight'), function (on) { keys.right = on; });
    $('tBoost').addEventListener('click', boost);

    var dragging = false, dragX = 0;
    cv.addEventListener('pointerdown', function (e) { dragging = true; dragX = e.clientX; });
    cv.addEventListener('pointermove', function (e) {
      if (!dragging) return;
      dragSteer = clamp((e.clientX - dragX) / 60, -1, 1);
    });
    function up() { dragging = false; dragSteer = 0; }
    cv.addEventListener('pointerup', up);
    cv.addEventListener('pointercancel', up);
    cv.addEventListener('pointerleave', up);
    window.addEventListener('resize', resize);
  }
  function bindHold(el, fn) {
    if (!el) return;
    el.addEventListener('pointerdown', function (e) { e.preventDefault(); fn(true); });
    ['pointerup', 'pointercancel', 'pointerleave'].forEach(function (k) {
      el.addEventListener(k, function () { fn(false); });
    });
  }
  function boost() {
    if (G.mode !== 'running' || G.boostCd > 0) return;
    G.boost = 1.1;
    G.boostCd = 3.2;
  }

  /* ------------------------------------------------------------------ */
  /* boot                                                                */
  /* ------------------------------------------------------------------ */
  function restore() {
    BEST = Object.assign({ score: 0, lap: 1, lapMs: 0 }, lsGet('best') || {});
    LIFE = Object.assign({ runs: 0, laps: 0, dist: 0, hits: 0, spawned: 0 },
                         lsGet('lifetime') || {});
    var q = Number(lsGet('quality'));
    if (q > 0) { QUALITY = q; $('quality').value = String(q); }
  }

  var lastTs = 0;
  function frame(ts) {
    var dt = lastTs ? Math.min(0.05, (ts - lastTs) / 1000) : 0.016;
    lastTs = ts;
    step(dt);
    paint();
    paintHud();
    if (G.mode === 'running') paintSide(false);
    requestAnimationFrame(frame);
  }

  function probeStore() {
    storeFetch('/v1/store?app=kart').then(function (r) { G.storeOk = r.ok; })
      .catch(function () { G.storeOk = false; });
  }

  function boot() {
    resize();
    restore();
    bindUI();
    renderFeeds();
    paintSide(true);
    paintHud();
    probeStore();
    loadHistory();
    setStatus('等待开始 · 路障全部由模型生成');
    requestAnimationFrame(frame);
    if (/[?&]debug/.test(location.search)) {
      window.__kart = {
        G: G, AGENT: AGENT,
        model: function () {
          return { id: MODEL_ID, base: apiBase() };
        },
        tris: function () { return nTri; },
        surf: surf, frameAt: frameAt
      };
    }
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
})();
