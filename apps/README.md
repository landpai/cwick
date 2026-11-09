# cwick apps

Each app is a self-contained H5 bundle under `apps/<id>/`; `manifest.json`
lists what the built-in app market shows. The server embeds this tree into the
binary (see `tools/embed_web.c`) and serves it at `/apps/...` on the same
origin as the OpenAI API, so an app needs no build step, no package manager and
no CORS configuration.

Layout:

```
apps/
  manifest.json          # the market catalog
  tictactoe/             # 九宫格井字棋
    app.json             # app metadata (kept next to the code)
    index.html           # entry
    app.css
    app.js
  kart/                  # 3D 泡泡卡丁车
    app.json
    index.html
    app.css
    app.js
```

Every app asks the model through `POST /v1/systemone`; the chat tab keeps
using `/v1/chat/completions`. Every request an app makes is same-origin —
there is no configurable external endpoint. The resident model is discovered
from `GET /v1/models` and re-discovered after a server-side switch, so the
decision adapter always follows the checkpoint that is actually loaded.

To add an app: create `apps/<id>/`, add its entry to `manifest.json`, then run
`make` (which builds and runs the C generator `tools/embed_web`) and restart
`cwick serve --web`.

## Where an app keeps its data

**Not** in `localStorage` alone. The server owns the databases in `~/.cwick/apps/`
(`--store DIR`, or `$CWICK_STORE`): **one file per app**, `<id>.db`, plus `auth.db`
for the server's own API keys. A `--store` path that exists as a file or ends in
`.db`/`.sqlite` still means one shared database — the old layout — and every
app's rows then live in that one file behind their `app_id`.

Run from a checkout, `--store ./apps` puts each app's database right next to its
code (`apps/tictactoe.db`, `apps/kart.db`). That is deliberate and safe:
`tools/embed_web.c` skips every database suffix when it walks `apps/`, so a
running server's files are never compiled into the binary, and `.gitignore`
keeps them out of a commit.

Three generic collections per app, all behind the same login as the model:

```
GET    /v1/store                          layout, per-app row counts, file size
GET    /v1/store?app=<id>                 that app's own database
GET    /v1/apps/<id>/state                every key            -> {"items":[...]}
GET    /v1/apps/<id>/state?key=K          one value            -> {"value":<json>}
PUT    /v1/apps/<id>/state                {"key":K,"value":V} or {"values":{...}}
DELETE /v1/apps/<id>/state?key=K
GET    /v1/apps/<id>/events?game_id=G&limit=N
POST   /v1/apps/<id>/events               {"game_id","seq","kind","payload"}
GET    /v1/apps/<id>/games?limit=N        newest first
PUT    /v1/apps/<id>/games                {"game_id","started_at","ended_at","outcome","moves"}
DELETE /v1/apps/<id>/games?game_id=G      the game and its events
```

Every value and payload must be JSON; it is stored as the exact text that
arrived and handed back verbatim, so an app never has to match a schema the
server invented. `<id>` is `[A-Za-z0-9_-]{1,64}` — checked before it reaches
sqlite — and limits are 1 MiB per value / payload and 5000 rows per read.

Why a server-side store rather than `localStorage`:

- "clear site data", a private window or a second browser profile would erase a
  match history the model produced;
- the phone that scanned the QR is a different browser and must see the same
  scoreboard and history;
- an app's history is worth keeping after the tab is gone.

`apps/tictactoe` uses all three collections: `state` for the scoreboard, `games`
for one row per match (`started_at`, `ended_at`, `outcome`, `moves`) and
`events` for one row per move (`kind` = `human`/`ai`, payload = board snapshot,
prompt with the per-cell tactical facts, the model's full probability
distribution as `probs`, raw model answer, parsed move, latency, usage). The
right-hand column's **历史回放** tab reads them back and replays a finished game
step by step.

`apps/kart` uses the same three: `games` for one row per run (`moves` = obstacles
the model placed), `events` for `kind` = `spawn`/`hit`/`lap`/`end` (payload = the
obstacle type, the position on the strip, the kart's speed, the model's latency,
its raw answer, plus the variation `seed`, pressure `focus` and decision
`entropy`) and `state` for the best score and a lifetime tally. Every spawn is a
model decision, so the replay shows what the model actually said for each
obstacle rather than a number from a generator.

## Which apps decide, and through which endpoint

Both hosted apps ask the model to choose, and both call `POST /v1/systemone`.
The chat tab is a different consumer: it uses `/v1/chat/completions`, because a
conversation is generated text, not a choice among options.

- **tic-tac-toe** — the human plays X first. Each machine move is a `choice`
  over the empty cells, and the question is explicitly *which cell gives O the
  greatest probability of winning*. The state carries the board and one line of
  tactical facts per empty cell (immediate win, forced block, winning replies,
  open lines). Those are facts, not a ranked list: the old wording — "win, else
  block, else centre, else corner" — was a local rule dressed up as a question
  and was removed. `fallbackMove()` remains, but only for a failed request.
- **kart** — a 12-way `choice` of obstacle kind and lane, asked in real time
  while the kart is moving. The question now asks for *maximum difficulty*; the
  previous "leave room for the player to react" was the opposite of the goal.

What stays local is what a letter cannot express: the kart's obstacle distance
is derived from speed (and tightens with the lap, above a reaction floor). The
model decides *what*, the app computes *where* — and an app never substitutes a
locally generated obstacle for a model decision.

Any app that caches a model id must handle a switch on the server side: a 400
whose text says the model is no longer served drops the cached id, rediscovers
the resident model and retries once. `GET /v1/systemone` answers which family,
prompt grammar and letter set are live, so a client can check rather than
infer.

## The win-probability readout in `apps/tictactoe`

`cellFacts()` computes, for each empty cell and each mark, whether the move
wins immediately, whether it blocks the opponent's immediate win, how many
winning replies remain, and how many lines are still open. It never ranks or
picks a cell — it is the same information a human counts, rendered into the
state and into each option's description. The endpoint's softmax over the
option letters is the model's distribution over "which cell wins most"; its
argmax is the move.

The right-hand column draws one bar per empty cell, highlights the cell
actually played, and shows the human's own tactical facts on their move, so
both sides of the game have a decision trail. The bars are a readout of one
prefill, **not** a Monte-Carlo win rate, and the heading says so: calling them
a simulated win percentage would be a claim the endpoint cannot make.

## The model decides the obstacles in `apps/kart`

`pumpAgent()` runs once per frame while racing and keeps at most one request in
flight and a two-deep queue of decisions, because a kart covers ~13 m in the
time a 320B model takes to answer — so the model is asked about a stretch of
track 45–95 m ahead rather than about the next frame.

The question is a 12-way `choice` at `/v1/systemone`: four obstacle kinds times
three lanes, one letter each. The model never writes a lane offset or a
distance — numbers it would have to generate token by token, and then be
repaired when it wrote prose around them. The distance is derived locally from
the kart's speed, which also keeps the road readable: the player sees where the
thing will be, not just what kind it is. The state is one line of English facts: lap, speed, hits, obstacles ahead,
spawned so far, difficulty level, which side of the strip is up, a variation
seed and a pressure point. Every byte is prefill on every obstacle, which is
why the option labels are short English and the Chinese names stay in the
sidebar.

**Real-time random decisions.** The readout is deterministic for a fixed
prompt, and a fixed A..L order would put the model's letter bias on the same
obstacle every time. Each request therefore Fisher-Yates shuffles the twelve
options (`criteria` is an ordered object on the server, so member order becomes
the letters), draws a fresh variation seed and one of six pressure points, and
asks for the hardest obstacle it can choose. The variation lives in the
*question*, never on the road: the seed, pressure point and the normalised
entropy of the resulting distribution are shown in the sidebar and written to
the sqlite `agent`/`spawn` events, so it is auditable rather than hidden.

**There is no local generator behind it.** This is the requirement, not an
accident: an empty queue means an empty road, the HUD says `模型离线 · 暂不放路障`,
and the sidebar's `已放路障` can never exceed the model's successful decisions.
The alternative — silently inventing obstacles when the model is slow — makes
the statistics a lie and hides a broken integration behind a playable game.

Three traps this app has already hit, all worth remembering:

- the model id must be the one the server advertises (`GET /v1/models`), never a
  literal — an invented name is a 400 and the model is never asked at all;
- `ensureModel()` returns the discovery promise from `resolveModel()`, which
  caches the in-flight `GET /v1/models` so several obstacles spawned in one
  frame share a single lookup. An empty answer must not be cached, or one cold
  start would leave the game without obstacles for the whole run. Failures like
  that are invisible on screen — the road is just empty for a moment — so the
  model state is exposed under `?debug`;
- a model id cached before a server-side switch is refused on *every* call
  ("model `x` is not served by this endpoint"). A 400 with that text now drops
  the id, rediscovers the resident model and retries once;
- one decision is one prefill, and the endpoint takes a generation slot: on this
  single-slot server an obstacle decision queues behind a chat turn. That is why
  the queue is only two deep and why an empty queue is shown as an empty road
  rather than papered over.

## The 3D renderer in `apps/kart`

No WebGL, no library: the kart is a small triangle rasteriser on a 2D canvas.
Each frame it pushes ~700 triangles into one buffer (track quads, curbs, centre
and edge lines, quarter-lap gates, obstacle boxes, the kart), sorts them
far-to-near and fills them. Three details carry the whole thing:

- **camera space is forward-positive.** `toCam()` dots a point with (right, up,
  forward) from the eye, so z > 0 means in front; the projection divides by z and
  a Sutherland–Hodgman clip against `z >= NEAR` trims the polygons that straddle
  the camera. An earlier version used the OpenGL convention (forward = −z) in the
  view matrix while the clip assumed +z, which culled the entire world;
- **normals are normalised before shading.** The cross product of a quad's edges
  has length = its area (a road segment is 8.8 m by 1.7 m, so |n| ≈ 15); using
  that directly as a brightness factor saturates every large quad to white,
  which is exactly how one build turned the strip into a white slab;
- **the camera's up vector is the surface normal**, so the horizon rolls with the
  strip instead of staying level with the world. That is what makes the flip
  readable rather than a floating camera over a twisted ribbon;
- **road texture must be indexed by world position, not by segment index.** The
  banding and centre dashes were picked with `i & 1` over the window the camera
  draws; because that window moves with the kart, the pattern moved with it and
  the road looked painted on the lens — 93 km/h and nothing on the surface said
  so. Snapping the segments to a world grid (`first = floor(u / du)`) makes the
  stripes stream past. A chase cam also needs a spring: with the eye an exact
  function of the kart's position, acceleration shows nothing; a few metres of
  lag, a 2 cm bob and a focal kick with speed are what "driving" looks like.

The track is a parametric Möbius strip, `S(u, v) = C(u) + A(u)·v` with
`A(u) = cos(u/2)·radial + sin(u/2)·up`, so after one lap the across direction is
reversed — one side, one surface, and the gates built in that frame arrive
upside down half a lap later.

The strip is drawn in two zones: the near ~1.3 rad at full detail (banding,
kerbs, dashes, painted start/twist bands) and the rest out to 4.4 rad as one
coarse quad per 0.08 rad plus a bright rim. Without the far zone a Möbius track
is indistinguishable from a banked ring — you only ever see the ten metres under
the kart. The rim is what carries the shape: a Möbius strip is recognised by its
single continuous edge.

A chase camera sits *inside* the twist, which is exactly why the app also draws a
small inset: the whole loop from outside the ring, with up-facing faces teal and
down-facing faces violet so the half-twist shows as a colour flip, plus the
kart's dot. Its projection is written separately from the main camera on
purpose — the main camera is a per-frame function of the kart and must not be
disturbed by a 140-pixel diagram. `?debug` exposes `window.__kart` (`G`, `AGENT`,
`tris()`, `frameAt`) so a headless run can tell "zero triangles" from "triangles
behind the camera".

## Sharing on the LAN

`cwick serve --lan` binds all interfaces (no auth/TLS; trusted Wi-Fi only) and
prints e.g. `LAN share: http://192.168.71.84:8001/apps/tictactoe/`. **Every app
here carries a 分享 / 二维码 button** — it calls `GET /v1/share?app=<id>` and
renders the answer through `GET /v1/share/qr?text=<url>`; scanning the PNG opens
that app standalone on a phone. Two things are worth knowing:

- **the `?app=` is not optional for a new app.** The server's default is
  `tictactoe`, so an app that forgets it will happily hand out a QR for somebody
  else's page. The id is charset-checked before it reaches `printf`.
- **`shareable:false` is a real answer**, returned with a `warning` when the
  listener is loopback-only or has no LAN IPv4. The sheet shows the reason and
  the `--lan` hint instead of a code that would not resolve.


With `--lan`, cwick forces single-user login, so an unauthenticated phone is
redirected to `/login?next=/apps/<id>/` and returns to the app after the
password — which is why every share note says the first open needs it. The apps
are responsive (tested at 390 px) and do not need the main console shell when
opened from that URL — the kart opens straight into its race because that is all
it does.
