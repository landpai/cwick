#!/usr/bin/env bash
#
# download_model.sh - fetch a whole model repository through hf-mirror.com
#
#   ./download_model.sh deepseek-ai/DeepSeek-V4.1-Flash-1017
#   ./download_model.sh deepseek-ai/DeepSeek-V4.1-Flash-1017 -o /Volumes/big/deepseek-v41
#   ./download_model.sh Qwen/Qwen3-32B --include '*.safetensors' '*.json' -j 3
#   ./download_model.sh deepseek-ai/DeepSeek-V4.1-Flash-1017 --list
#
# Why this exists instead of `hf download`:
#   * no Python required - curl plus bash, so it runs on a bare machine;
#   * resumable at any point, including across reboots, using the same
#     `<file>.incomplete` convention huggingface_hub uses, so partial files are
#     interchangeable between the two tools;
#   * built for files above 100 GB: no 2 GiB/4 GiB assumptions anywhere, the
#     filesystem is checked (FAT32 caps a single file at 4 GiB) and `ulimit -f`
#     is checked (a 2 GB limit kills a 300 GB download two hours in);
#   * one aggregate progress line - percent of the whole repository, bytes,
#     rate, ETA and file count - instead of one curl bar per shard. Per-shard
#     bars answer "how is shard 12 doing", never "when can I load this model",
#     and with -j above 1 they overwrite each other. `--no-progress` turns it
#     off for a log file;
#   * every transfer is verified against the size the API reports (and a
#     transfer that starts from byte 0 may overrule a stale listing), and LFS
#     blobs can be verified against their SHA-256 with --verify-sha.
#
# The tree comes from `${HF_ENDPOINT}/api/models/<repo>/tree/...` and each file
# from `${HF_ENDPOINT}/<repo>/resolve/<rev>/<path>`. Both answer with a redirect
# to the CDN; curl follows it, and on the hop it drops the Authorization header
# by itself because the host changes.
#
# Exit status: 0 all selected files are complete, 1 something failed or the
# repository could not be listed, 130 interrupted (re-run to resume).

set -u

# Running it as `sh download_model.sh` is a common reflex; re-exec under bash,
# because everything below uses arrays and `local`.
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi

PROG=${0##*/}

# ---------------------------------------------------------------------------
# defaults
# ---------------------------------------------------------------------------

ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
REPO=''
REVISION='main'
REPO_TYPE='model'
OUT_DIR=''
TOKEN=${HF_TOKEN:-}
JOBS=1
RETRIES=5
INCLUDE=()
EXCLUDE=()
FORCE=0
LIST_ONLY=0
DRY_RUN=0
VERIFY_SHA=0
DISK_CHECK=1
PROGRESS=1
WRITER_CHECK=1
CURL_PID=""
LOCK=''

usage() {
    cat <<'EOF'
Download a complete model repository through hf-mirror.com.

Usage:
  download_model.sh <owner>/<repo> [options]
  download_model.sh --repo <owner>/<repo> [options]

Options:
  -o, --output-dir DIR   Where to write the files. Default: ./<repo name>
      --local-dir DIR    Alias of --output-dir (the huggingface-cli spelling).
  -r, --revision REV     Branch, tag or commit. Default: main
  -t, --type TYPE        model (default), dataset or space.
      --include GLOB ...  Only download paths matching one of these shell
                          globs, e.g. --include '*.safetensors' '*.json'.
      --exclude GLOB ...  Skip paths matching one of these globs.
  -j, --jobs N           Files to download at once. Default: 1 (one progress
                          bar, sequential; on a fast link use 2-4).
      --retries N        Attempts per file before giving up. Default: 5
      --token TOKEN      Hugging Face token. Default: $HF_TOKEN, else
                          ~/.cache/huggingface/token if present.
      --endpoint URL     Mirror base URL. Default: $HF_ENDPOINT or
                          https://hf-mirror.com
      --list             Print every file in the repository with its size and
                          exit (include/exclude are ignored).
      --dry-run          Print what this invocation would download, after
                          include/exclude, and exit.
      --verify-sha       Verify each LFS blob against its SHA-256 after
                          download. Reads the whole file again, so it is off by
                          default; expect minutes per 100 GB.
      --force            Replace files whose size does not match the remote.
      --no-disk-check    Skip the free-space estimate.
      --no-progress      Do not draw the aggregate progress line; print only
                          the per-file result lines (for a log file, or cron).
      --no-writer-check  Start even if another program is writing into the
                          output directory (it would corrupt the resume).
  -h, --help             This text.

Examples:
  # the whole repository: 83 files, 475 GiB (two shards are 94.6 GiB each)
  ./download_model.sh deepseek-ai/DeepSeek-V4.1-Flash-1017

  # weights only, three at a time, onto an external SSD
  ./download_model.sh deepseek-ai/DeepSeek-V4.1-Flash-1017 \
      -o /Volumes/ssd/models/deepseek-v41 --include '*.safetensors' '*.json' -j 3

  # what is in there, and how much would it cost
  ./download_model.sh deepseek-ai/DeepSeek-V4.1-Flash-1017 --list

  # a gated repository with a token
  HF_TOKEN=hf_xxx ./download_model.sh meta-llama/Llama-3.1-8B

Interrupting is safe: every file lives in <name>.incomplete until it is
complete, and re-running the same command continues from the byte it reached.
Only one run may write into a given output directory at a time; a second one
exits immediately instead of appending to the same .incomplete files. A partial
file touched in the last minute is also checked against the process table, so a
writer that was killed (terminal closed, machine slept) no longer blocks the
resume.
EOF
}

# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

die() { printf '%s: %s\n' "$PROG" "$*" >&2; exit 1; }
warn() { printf '%s: warning: %s\n' "$PROG" "$*" >&2; }
have() { command -v "$1" >/dev/null 2>&1; }

# 1536 -> 1.50KiB. Binary units, because that is what df and the API speak.
human() {
    awk -v b="${1:-0}" 'BEGIN{
        n = split("B KiB MiB GiB TiB PiB", u, " "); i = 1
        while (b >= 1024 && i < n) { b /= 1024; i++ }
        if (i == 1) printf "%d%s", b, u[i]; else printf "%.2f%s", b, u[i]
    }'
}

# st_size is spelled differently on macOS and Linux; -f%z wins if it works.
fsize() {
    if [ ! -e "$1" ]; then printf '0'; return 0; fi
    if stat -f %z "$1" >/dev/null 2>&1; then
        stat -f %z "$1"
    else
        stat -c %s "$1"
    fi
}

now() { date +%s; }

# ---------------------------------------------------------------------------
# progress
# ---------------------------------------------------------------------------
#
# The bar is aggregate: percent of the selected set, bytes, rate and ETA. A
# worker never draws it. It registers the file it is writing in
# $TMP/active/<id> ("<size>\t<path>") and drops its final on-disk byte count in
# $TMP/done/<id>, so the ticker - the only writer of the bar - can sum both and
# -j 4 cannot interleave two half-drawn lines. Everything the ticker reads is a
# file, which is also why it can live in a subshell.

PROG_TTY=0          # 1: redraw one line in place, 0: a line every 15 s
PROG_TOTAL=0        # bytes in the selected set (0 if the listing had no sizes)
PROG_FILES=0        # files in the selected set
PROG_SPEED=0        # bytes/s, refreshed every 5 s
PROG_S_T=0          # time of the last speed sample
PROG_S_B=0          # transferred bytes at the last speed sample
PROG_LAST=0         # time of the last non-tty print (0: nothing printed yet)
PROG_FORCE=0        # 1: print this frame even if the 15 s window has not passed
TICKER=''

fmt_eta() { # seconds -> 2d05h / 4h12m / 7m03s / 45s
    awk -v s="${1:-0}" 'BEGIN{
        if (s <= 0 || s > 8640000) { printf "--"; exit }
        d = int(s / 86400); s -= d * 86400
        h = int(s / 3600);  s -= h * 3600
        m = int(s / 60);    s -= m * 60
        if (d > 0)      printf "%dd%02dh", d, h
        else if (h > 0) printf "%dh%02dm", h, m
        else if (m > 0) printf "%dm%02ds", m, s
        else            printf "%ds", s
    }'
}

# Sets PB_DONE (bytes of finished files), PB_NF (finished count), PB_LIVE
# (bytes in flight), PB_BASE (what those in-flight files had on disk when their
# worker started) and PB_CUR (a file being written, for the line).
#
# Two different quantities are needed and they are not the same number:
#   PB_DONE + PB_LIVE   - how much of the repository is on disk, which is what
#                         the percent means (a run that skips 35 finished
#                         shards is already 50% of the way there);
#   PB_XFER             - how many bytes this run actually pulled off the
#                         network, which is the only thing a rate or an ETA may
#                         be computed from. Mixing them up shows 36 GiB/s and
#                         "eta 6s" while 35 shards are being skipped.
prog_sample() {
    local f sz part cur base
    PB_DONE=0; PB_NF=0; PB_LIVE=0; PB_BASE=0; PB_CUR=''
    for f in "$TMP"/done/*; do
        [ -e "$f" ] || continue
        sz=''
        read -r sz < "$f" || :
        PB_DONE=$((PB_DONE + ${sz:-0}))
        PB_NF=$((PB_NF + 1))
    done
    for f in "$TMP"/active/*; do
        [ -e "$f" ] || continue
        sz=''; part=''
        IFS=$'\t' read -r sz part < "$f" || :
        [ -n "${part:-}" ] || continue
        cur=$(fsize "$part")
        PB_LIVE=$((PB_LIVE + cur))
        PB_BASE=$((PB_BASE + ${sz:-0}))
        if [ -z "$PB_CUR" ]; then
            PB_CUR=${part##*/}
            PB_CUR=${PB_CUR%.incomplete}
        fi
    done
    PB_XFER=0
    if [ -s "$TMP/xfer" ]; then
        PB_XFER=$(awk '{s += $1} END{printf "%d", s}' "$TMP/xfer")
    fi
    PB_XFER=$((PB_XFER + PB_LIVE - PB_BASE))
    [ "$PB_XFER" -lt 0 ] && PB_XFER=0
}

prog_begin() { # id part
    printf '%s\t%s\n' "$(fsize "$2")" "$2" > "$TMP/active/$1"
}
# With a byte count the file counts towards the bar; without one it only stops
# being "in flight", so a failure leaves the bar short of 100% instead of
# claiming a file that is not there. The delta is what this run transferred for
# it, and goes to the rate accumulator.
prog_end() { # id [bytes] [delta]
    rm -f "$TMP/active/$1"
    [ $# -ge 2 ] && printf '%s\n' "${2:-0}" > "$TMP/done/$1"
    [ $# -ge 3 ] && [ "${3:-0}" -gt 0 ] && printf '%s\n' "$3" >> "$TMP/xfer"
    return 0
}

# One frame. Called once a second by the ticker, and once more after it stops
# so the bar cannot be left showing 97%.
prog_render() {
    local t cur pct fill empty bar speed eta line
    t=$(now)
    prog_sample
    cur=$((PB_DONE + PB_LIVE))
    [ "$PROG_TOTAL" -gt 0 ] || PROG_TOTAL=$cur
    if [ "$PROG_TOTAL" -gt 0 ]; then
        pct=$((cur * 1000 / PROG_TOTAL))
    else
        pct=0
    fi
    [ "$pct" -gt 1000 ] && pct=1000

    # Rate over a 5 s window, on bytes this run transferred: an average from
    # byte 0 hides that the link just dropped to 200 KB/s, and a 1 s window is
    # too noisy to read.
    if [ "$PROG_S_T" -eq 0 ]; then
        PROG_S_T=$t; PROG_S_B=$PB_XFER
    elif [ "$t" -ge $((PROG_S_T + 5)) ]; then
        if [ "$PB_XFER" -ge "$PROG_S_B" ] && [ "$t" -gt "$PROG_S_T" ]; then
            PROG_SPEED=$(( (PB_XFER - PROG_S_B) / (t - PROG_S_T) ))
        fi
        PROG_S_T=$t; PROG_S_B=$PB_XFER
    fi
    speed=$PROG_SPEED
    if [ "$speed" -le 0 ] && [ "$PB_XFER" -gt 0 ] && [ "${START:-0}" -gt 0 ] && [ "$t" -gt "$START" ]; then
        speed=$(( PB_XFER / (t - START) ))   # first frame, and the final one
    fi
    if [ "$speed" -gt 0 ] && [ "$PROG_TOTAL" -gt "$cur" ]; then
        eta=$(fmt_eta $(( (PROG_TOTAL - cur) / speed )))
    else
        eta='--'
    fi

    fill=$((pct * 24 / 1000))
    bar=$(printf '%*s' "$fill" '' | tr ' ' '=')
    empty=$(printf '%*s' $((24 - fill)) '' | tr ' ' '-')
    line=$(printf '[%s%s] %3d.%d%%  %s/%s  %s/s  eta %s  %s/%s' \
        "$bar" "$empty" $((pct / 10)) $((pct % 10)) \
        "$(human "$cur")" "$(human "$PROG_TOTAL")" "$(human "$speed")" \
        "$eta" "$PB_NF" "$PROG_FILES")
    [ -n "$PB_CUR" ] && line="$line  ${PB_CUR:0:28}"

    if [ "$PROG_TTY" = 1 ]; then
        printf '\r\033[K%s' "$line"
    elif [ "$PROG_FORCE" = 1 ] || [ "$PROG_LAST" -eq 0 ] || [ "$t" -ge $((PROG_LAST + 15)) ]; then
        printf '%s\n' "$line"
        PROG_LAST=$t
    fi
    return 0
}

# Per-file lines must not land in the middle of the bar.
emit() {
    [ "$PROG_TTY" = 1 ] && printf '\r\033[K'
    printf "$@"
}

prog_init() {
    local size path oid
    PROG_TOTAL=0
    PROG_FILES=0
    while IFS=$'\t' read -r size path oid; do
        [ -n "${path:-}" ] || continue
        PROG_TOTAL=$((PROG_TOTAL + ${size:-0}))
        PROG_FILES=$((PROG_FILES + 1))
    done < "$TMP/selected"
}

prog_start() {
    [ "$PROGRESS" = 1 ] || return 0
    [ -t 1 ] && PROG_TTY=1
    PROG_LAST=$(now)   # non-tty: the first line comes after 15 s, not at 0%
    if [ "$PROG_TTY" = 1 ]; then printf '\033[?25l'; fi  # hide cursor
    ( while [ ! -e "$TMP/stop" ]; do
          prog_render
          sleep 1
      done ) &
    TICKER=$!
    return 0
}

prog_stop() {
    [ -n "${TICKER:-}" ] || return 0
    : > "$TMP/stop"
    # Not killed, just told to stop: the ticker re-checks once a second, and a
    # SIGTERM would make bash print "Terminated: 15 ( while ..." over the
    # summary. One second of patience buys a clean exit.
    wait "$TICKER" 2>/dev/null || :
    TICKER=''
    # Always leave one last line behind. Without it a run that finishes inside
    # the first second - everything already on disk, a repo of small files -
    # writes no progress at all, and the log ends on the per-file lines.
    PROG_FORCE=1
    prog_render
    PROG_FORCE=0
    if [ "$PROG_TTY" = 1 ]; then
        printf '\n\033[?25h'                            # show cursor again
    fi
    return 0
}

# Percent-encode everything except unreserved bytes and '/', so a path with
# spaces, '+', '#', '%' or CJK characters still produces one valid URL. The
# loop runs under LC_ALL=C, which makes ${s:i:1} yield single bytes.
url_escape() {
    local s=$1 out='' c i
    local LC_ALL=C
    for (( i = 0; i < ${#s}; i++ )); do
        c=${s:i:1}
        case $c in
            [A-Za-z0-9._~/-]) out=$out$c ;;
            \') out="$out%27" ;;
            *) printf -v c '%%%02X' "'$c"; out=$out$c ;;
        esac
    done
    printf '%s' "$out"
}

# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

api_get() { # url body headers
    if [ -n "$TOKEN" ]; then
        curl -fsSL --globoff --connect-timeout 30 --max-time 180 \
            -H "Authorization: Bearer $TOKEN" -D "$3" -o "$2" "$1"
    else
        curl -fsSL --globoff --connect-timeout 30 --max-time 180 \
            -D "$3" -o "$2" "$1"
    fi
}

# One transfer. Writes the HTTP status into $codefile (stdout would be inside a
# command-substitution subshell, which would leave curl orphaned on Ctrl-C) and
# returns curl's exit status. -C - asks the server to continue from whatever is
# already on disk; because the offset is re-derived from the file size on every
# call, retrying is always safe even if the previous attempt died mid-write.
dl_curl() { # part url codefile
    local part=$1 url=$2 codefile=$3
    # -sS rather than --progress-bar: the aggregate line above is the progress
    # display, and curl's bar is unreadable the moment stdout is not a tty (it
    # writes the raw frames into the log) or -j is above 1.
    if [ -n "$TOKEN" ]; then
        curl -fL --globoff -sS --continue-at - \
            --connect-timeout 30 --speed-limit 2048 --speed-time 60 \
            -H "Authorization: Bearer $TOKEN" \
            -o "$part" -w '%{http_code}' "$url" > "$codefile"
    else
        curl -fL --globoff -sS --continue-at - \
            --connect-timeout 30 --speed-limit 2048 --speed-time 60 \
            -o "$part" -w '%{http_code}' "$url" > "$codefile"
    fi
}

# Authoritative length of one file, straight from a one-byte range request.
# The mirror answers with 307 + "content-range: bytes 0-0/<total>", a plain
# CDN with 206 + the same header, and a server that ignores ranges with 200 +
# content-length - three spellings of the same number. Empty output means none
# of them was usable. One tiny request, so it is only asked when the listing
# and the bytes on disk disagree.
remote_size() { # url headers-file
    if [ -n "$TOKEN" ]; then
        curl -fsSL --globoff -r 0-0 --connect-timeout 30 --max-time 120 \
            -H "Authorization: Bearer $TOKEN" -D "$2" -o /dev/null "$1" 2>/dev/null
    else
        curl -fsSL --globoff -r 0-0 --connect-timeout 30 --max-time 120 \
            -D "$2" -o /dev/null "$1" 2>/dev/null
    fi
    [ -s "$2" ] || return 0
    tr -d '\r' < "$2" | awk '
        /^HTTP\// { code = $2; next }
        tolower($1) == "content-range:" {
            n = split($0, a, "/")
            if (a[n] ~ /^[0-9]+$/) cr = a[n]
            next
        }
        tolower($1) == "content-length:" {
            if (code ~ /^2/ && cl == "") cl = $2
            next
        }
        END {
            if (cr != "") print cr
            else if (cl + 0 > 0) print cl
        }'
}

# ---------------------------------------------------------------------------
# repository listing
# ---------------------------------------------------------------------------

JQ=''
PY=''
pick_json_tools() {
    if have jq; then JQ=$(command -v jq); return 0; fi
    if have python3; then PY=$(command -v python3); return 0; fi
    die "need jq or python3 to read the repository file list"
}

# Emit "size<TAB>path<TAB>oid" for every file in one page of the tree API.
# For LFS files the useful size is lfs.size; .size can be the pointer itself.
parse_page() {
    if [ -n "$JQ" ]; then
        "$JQ" -r '
            .[]?
            | select(.type == "file")
            | [ (.lfs.size // .size // 0), .path, (.lfs.oid // "") ]
            | @tsv' "$1"
    else
        "$PY" - "$1" <<'PY'
import json, sys
try:
    with open(sys.argv[1], "rb") as fp:
        entries = json.load(fp)
except Exception:
    sys.exit("unreadable JSON from the tree API")
for e in entries:
    if not isinstance(e, dict) or e.get("type") != "file":
        continue
    lfs = e.get("lfs") or {}
    size = lfs.get("size") or e.get("size") or 0
    print("%d\t%s\t%s" % (size, e.get("path", ""), lfs.get("oid") or ""))
PY
    fi
}

# The Link header carries the cursor for the next page as an absolute URL on
# huggingface.co. Rewrite it onto the mirror, otherwise a repository with more
# than one page of files silently loses its second half (it is truncated at
# 1000 entries per page, which a 300 GB checkpoint exceeds).
next_page_url() { # headers
    local n
    n=$(tr -d '\r' < "$1" \
        | grep -o '<[^>]*>; rel="next"' \
        | head -1 \
        | sed -e 's/^<//' -e 's/>; rel="next"$//')
    [ -n "$n" ] || return 0
    n=$(printf '%s' "$n" | sed -E 's#^[A-Za-z][A-Za-z0-9+.-]*://[^/]+##')
    printf '%s%s' "$ENDPOINT" "$n"
}

list_repo() { # out.tsv
    local out=$1 url pages=0 api
    case "$REPO_TYPE" in
        model)   api="models" ;;
        dataset) api="datasets" ;;
        space)   api="spaces" ;;
        *) die "unknown --type: $REPO_TYPE" ;;
    esac
    url="$ENDPOINT/api/$api/$REPO/tree/$REVISION?recursive=true&expand=true"
    : > "$out"
    while [ -n "$url" ]; do
        pages=$((pages + 1))
        if ! api_get "$url" "$TMP/page.json" "$TMP/page.hdr"; then
            die "cannot list $REPO at '$REVISION' (HTTP error). Check the repo id and the revision, and use --token if it is gated."
        fi
        parse_page "$TMP/page.json" >> "$out" || die "cannot parse the file list"
        url=$(next_page_url "$TMP/page.hdr")
    done
    [ -s "$out" ] || die "$REPO has no files at '$REVISION'"
}

# ---------------------------------------------------------------------------
# selection, disk, filesystem
# ---------------------------------------------------------------------------

match_any() { # path patterns...
    local p=$1 pat
    shift
    for pat in "$@"; do
        case $p in $pat) return 0 ;; esac
    done
    return 1
}

select_files() { # full.tsv selected.tsv
    local src=$1 dst=$2 size path oid n=0
    : > "$dst"
    while IFS=$'\t' read -r size path oid; do
        [ -n "${path:-}" ] || continue
        if [ ${#INCLUDE[@]} -gt 0 ] && ! match_any "$path" "${INCLUDE[@]}"; then
            continue
        fi
        if [ ${#EXCLUDE[@]} -gt 0 ] && match_any "$path" "${EXCLUDE[@]}"; then
            continue
        fi
        printf '%s\t%s\t%s\n' "$size" "$path" "$oid" >> "$dst"
        n=$((n + 1))
    done < "$src"
    printf '%s\n' "$n" > "$TMP/count"
}

print_table() { # selected.tsv
    local size path oid total=0 n=0
    printf '%-12s %s\n' SIZE PATH
    while IFS=$'\t' read -r size path oid; do
        [ -n "${path:-}" ] || continue
        printf '%-12s %s\n' "$(human "$size")" "$path"
        total=$((total + size))
        n=$((n + 1))
    done < "$1"
    printf '\n%d file(s), %s total\n' "$n" "$(human "$total")"
}

# FAT32/exFAT cannot hold a single file above 4 GiB, which turns a 300 GB
# checkpoint into a failure four hours in. Check before starting, not after.
check_filesystem() { # dir
    local dir=$1 fs='' big dev
    big=$(awk -F'\t' 'BEGIN{m=0} {if ($1 + 0 > m) m = $1 + 0} END{print m + 0}' "$TMP/selected")
    [ "${big:-0}" -gt 4294967296 ] || return 0
    if stat -f -c %T "$dir" >/dev/null 2>&1; then
        fs=$(stat -f -c %T "$dir")
    else
        dev=$(df -P "$dir" 2>/dev/null | awk 'NR==2{print $1}')
        if [ -n "$dev" ]; then
            fs=$(mount 2>/dev/null | awk -v d="$dev" '
                $1 == d { for (i = 1; i <= NF; i++)
                    if ($i ~ /^\(/) { gsub(/[()]/, "", $i); sub(/,.*/, "", $i); print $i; exit } }')
        fi
    fi
    case "$fs" in
        msdos|vfat|fat|fat32|exfat)
            warn "target filesystem is $fs: a single file cannot exceed 4 GiB, and this repository has one at $(human "$big"). Use APFS or ext4 instead."
            ;;
    esac
}

check_ulimit() {
    local lim shown
    lim=$(ulimit -f 2>/dev/null) || return 0
    [ -n "$lim" ] || return 0
    [ "$lim" = "unlimited" ] && return 0
    # -f is counted in 512-byte blocks on macOS, 1024-byte blocks on Linux;
    # the smaller block size makes the warning conservative.
    shown=$(human $((lim * 512)))
    warn "ulimit -f is $lim blocks (~$shown); a larger file will be cut off. Run 'ulimit -f unlimited' first."
}

check_disk() { # dir
    local size path oid dest cur need=0 avail
    while IFS=$'\t' read -r size path oid; do
        [ -n "${path:-}" ] || continue
        [ "$size" -gt 0 ] || continue
        dest="$OUT_DIR/$path"
        cur=0
        [ -f "$dest" ] && cur=$(fsize "$dest")
        if [ -f "$dest.incomplete" ] && [ "$(fsize "$dest.incomplete")" -gt "$cur" ]; then
            cur=$(fsize "$dest.incomplete")
        fi
        if [ "$size" -gt "$cur" ]; then need=$((need + size - cur)); fi
    done < "$TMP/selected"
    [ "$need" -gt 0 ] || return 0
    avail=$(df -Pk "$1" 2>/dev/null | awk 'NR==2{print $4}')
    [ -n "$avail" ] || return 0
    avail=$((avail * 1024))
    if [ "$need" -gt "$avail" ]; then
        if [ "$DISK_CHECK" = 1 ]; then
            die "needs $(human "$need"), only $(human "$avail") free on $1. Free space, choose another -o, or pass --no-disk-check."
        fi
        warn "needs $(human "$need") but only $(human "$avail") free on $1"
    fi
    printf 'space: %s to fetch, %s free\n' "$(human "$need")" "$(human "$avail")"
}

# ---------------------------------------------------------------------------
# one file
# ---------------------------------------------------------------------------

sha256_of() {
    if have sha256sum; then sha256sum < "$1" | awk '{print $1}'
    else shasum -a 256 < "$1" | awk '{print $1}'; fi
}

# Returns 0 when the file was fetched, 1 when it failed, 2 when it was already
# there, so the summary can tell "skipped" from "downloaded".
download_one() { # size path oid id
    local size=$1 path=$2 oid=$3 id=${4:-0}
    local dest="$OUT_DIR/$path" part="$OUT_DIR/$path.incomplete"
    local codefile="$TMP/code.$id"
    local url cur now_b before base attempt=1 rc=0 code='' t0 t1

    [ -f "$TMP/abort" ] && return 1
    mkdir -p "$(dirname "$dest")"

    if [ -f "$dest" ]; then
        cur=$(fsize "$dest")
        if [ "$size" -le 0 ] || [ "$cur" -eq "$size" ]; then
            emit 'skip    %s\n' "$path"
            prog_end "$id" "$cur"
            return 2
        fi
        if [ "$FORCE" = 1 ]; then
            rm -f "$dest"
        elif [ "$cur" -lt "$size" ]; then
            # A previous run (or huggingface_hub) left a short file: keep those
            # bytes and finish it instead of starting the 40 GB over.
            if [ -f "$part" ] && [ "$(fsize "$part")" -ge "$cur" ]; then
                rm -f "$dest"
            else
                mv -f "$dest" "$part"
            fi
        else
            printf 'bad     %s: local %s > remote %s; use --force\n' \
                "$path" "$(human "$cur")" "$(human "$size")"
            return 1
        fi
    fi

    # A leftover bigger than the remote file (a different revision, or a mirror
    # that answered a range request with the whole body) can never be resumed;
    # drop it now instead of letting every attempt fail with 416.
    cur=$(fsize "$part")
    if [ "$size" -gt 0 ] && [ "$cur" -gt "$size" ]; then
        warn "$path: partial file is larger than the remote one ($(human "$cur") > $(human "$size")), starting over"
        rm -f "$part"
    fi

    url="$ENDPOINT/$RESOLVE_PREFIX/resolve/$REVISION/$(url_escape "$path")"
    t0=$(now)
    # Bytes already on disk for this file: the difference at the end is what
    # this run contributed, which is what the rate and the ETA are made of.
    base=$(fsize "$part")
    prog_begin "$id" "$part"

    while [ "$attempt" -le "$RETRIES" ]; do
        before=$(fsize "$part")
        if [ "$before" -gt 0 ] && [ "$size" -gt 0 ]; then
            emit 'resume  %s at %s / %s\n' "$path" "$(human "$before")" "$(human "$size")"
        fi
        # curl runs as a real job (not inside $(...)) so that on_int can reach
        # it: a command substitution would make curl a grandchild, survive the
        # trap, and quietly finish the download after the script has gone.
        dl_curl "$part" "$url" "$codefile" &
        CURL_PID=$!
        wait "$CURL_PID"
        rc=$?
        CURL_PID=''
        code=''
        [ -s "$codefile" ] && code=$(cat "$codefile")

        if [ "$rc" -eq 0 ]; then
            now_b=$(fsize "$part")
            if [ "$size" -le 0 ]; then
                # The listing gave no size for this path: whatever arrived is
                # the file.
                size=$now_b
                if [ "$size" -gt 0 ]; then break; fi
                emit 'short   %s: empty transfer, retrying\n' "$path" >&2
                attempt=$((attempt + 1))
                continue
            fi
            [ "$now_b" -eq "$size" ] && break

            # The listing and the transfer disagree. Ask the server what the
            # file really is before spending five attempts on the wrong number
            # - and do it once per file, not once per attempt.
            if [ ! -e "$TMP/probed.$id" ]; then
                : > "$TMP/probed.$id"
                real=$(remote_size "$url" "$TMP/probe.$id")
                if [ -n "${real:-}" ] && [ "$real" -gt 0 ] && [ "$real" -ne "$size" ]; then
                    warn "$path: the listing said $(human "$size") ($size bytes), the server serves $(human "$real") ($real bytes); using the server's length"
                    size=$real
                    if [ "$now_b" -eq "$size" ]; then break; fi
                    if [ "$now_b" -gt "$size" ]; then rm -f "$part"; fi
                    attempt=$((attempt + 1))
                    continue
                fi
            fi

            if [ "$now_b" -gt "$size" ]; then
                if [ "$before" -eq 0 ]; then
                    # A transfer that started at byte 0 defines the length; the
                    # listing can lag behind a repository update. Adopt it
                    # instead of failing five times on a file that is whole.
                    warn "$path: server sent $(human "$now_b"), the listing said $(human "$size"); using the server's length"
                    size=$now_b
                    break
                fi
                # More bytes than the file should have, after resuming: the
                # offset no longer lines up (a second writer appended to the
                # same .incomplete, or a mirror ignored the range and sent the
                # whole body). Only a fresh start can tell what the file is.
                warn "$path: $(human "$now_b") ($now_b bytes) after resuming at $(human "$before"), but $(human "$size") ($size bytes) expected; restarting this file from 0"
                rm -f "$part"
                attempt=$((attempt + 1))
                continue
            fi
            emit 'short   %s: got %s of %s, retrying\n' \
                "$path" "$(human "$now_b")" "$(human "$size")" >&2
            attempt=$((attempt + 1))
            continue
        fi

        case "$rc" in
            33)
                # "HTTP server does not seem to support byte ranges", or 416
                # because the file is already whole. A full file is done; a
                # partial one has to restart from zero, here and only here.
                now_b=$(fsize "$part")
                if [ "$size" -gt 0 ] && [ "$now_b" -eq "$size" ]; then
                    rc=0
                    break
                fi
                if [ "$before" -gt 0 ]; then
                    warn "$path: mirror ignored the range request, restarting this file"
                    rm -f "$part"
                fi
                ;;
            22)
                case "$code" in
                    416)
                        now_b=$(fsize "$part")
                        if [ "$size" -gt 0 ] && [ "$now_b" -eq "$size" ]; then
                            rc=0
                            break
                        fi
                        warn "$path: server rejected the resume offset (416), starting over"
                        rm -f "$part"
                        ;;
                    401|403|451)
                        # Every other file would fail the same way: leave a flag
                        # so the rest of the queue stops instead of hammering.
                        printf 'the mirror answered HTTP %s; pass --token (and accept the licence on the site for a gated repository)\n' \
                            "$code" > "$TMP/abort"
                        emit 'failed  %s: HTTP %s, the repository is gated or private\n' \
                            "$path" "$code" >&2
                        prog_end "$id"
                        return 1
                        ;;
                    404)
                        emit 'failed  %s: HTTP 404, not at revision %s\n' \
                            "$path" "$REVISION" >&2
                        prog_end "$id"
                        return 1
                        ;;
                esac
                ;;
            23)
                printf 'cannot write to %s: disk full, or the volume went away\n' \
                    "$OUT_DIR" > "$TMP/abort"
                emit 'failed  %s: write error\n' "$path" >&2
                prog_end "$id"
                return 1
                ;;
        esac

        attempt=$((attempt + 1))
        if [ "$attempt" -le "$RETRIES" ]; then
            emit 'retry   %s: curl exit %s (HTTP %s), attempt %d/%d\n' \
                "$path" "$rc" "${code:-?}" "$attempt" "$RETRIES" >&2
            sleep $((attempt * 2 > 20 ? 20 : attempt * 2))
        fi
    done

    if [ "$rc" -ne 0 ]; then
        emit 'failed  %s after %d attempt(s); %s is kept, re-run to resume\n' \
            "$path" "$RETRIES" "$(human "$(fsize "$part")")" >&2
        prog_end "$id"
        return 1
    fi

    if [ "$size" -gt 0 ] && [ "$(fsize "$part")" -ne "$size" ]; then
        emit 'failed  %s: final size %s != %s\n' \
            "$path" "$(human "$(fsize "$part")")" "$(human "$size")" >&2
        prog_end "$id"
        return 1
    fi

    if [ "$VERIFY_SHA" = 1 ] && [ -n "$oid" ]; then
        emit 'verify  %s\n' "$path"
        if [ "$(sha256_of "$part")" != "$oid" ]; then
            mv -f "$part" "$part.bad"
            emit 'failed  %s: SHA-256 mismatch; the bad file is at %s.bad\n' \
                "$path" "$part" >&2
            prog_end "$id"
            return 1
        fi
    fi

    mv -f "$part" "$dest"
    t1=$(now)
    emit 'done    %s  %s in %ss\n' "$path" "$(human "$(fsize "$dest")")" "$((t1 - t0))"
    printf '%s\n' "$(fsize "$dest")" >> "$TMP/bytes"
    prog_end "$id" "$size" "$(( $(fsize "$dest") - base ))"
    return 0
}

run_sequential() {
    local size path oid failed=0 rc idx=0
    while IFS=$'\t' read -r size path oid; do
        [ -n "${path:-}" ] || continue
        idx=$((idx + 1))
        download_one "$size" "$path" "$oid" "$idx"
        rc=$?
        [ "$rc" -eq 1 ] && failed=$((failed + 1))
    done < "$TMP/selected"
    printf '%s\n' "$failed" > "$TMP/failed"
}

# bash 3.2 has no `wait -n`, and `jobs` cannot be called inside $(...), so the
# pool is a job count written to a file plus a short poll. Each worker drops a
# marker on failure because a background job's status is gone once reaped.
#
# Neither the pool nor the final reap may use a bare `wait`: the progress ticker
# is a background job too, and it only exits after --stop exists, which happens
# after this function returns. `wait` would therefore wait for itself.
run_parallel() {
    local size path oid idx=0 running=0 f n=0 extra=0
    mkdir -p "$TMP/fail"
    : > "$TMP/workers"
    [ -n "${TICKER:-}" ] && extra=1
    while IFS=$'\t' read -r size path oid; do
        [ -n "${path:-}" ] || continue
        idx=$((idx + 1))
        (
            download_one "$size" "$path" "$oid" "$idx"
            [ $? -eq 1 ] && printf 'x\n' > "$TMP/fail/$idx"
        ) &
        printf '%s\n' "$!" >> "$TMP/workers"
        while :; do
            jobs -rp > "$TMP/jobs.txt" 2>/dev/null || :
            running=$(wc -l < "$TMP/jobs.txt" | tr -d ' ')
            [ "${running:-0}" -ge $((JOBS + extra)) ] || break
            sleep 0.2
        done
    done < "$TMP/selected"
    while read -r f; do
        [ -n "$f" ] && wait "$f" 2>/dev/null
    done < "$TMP/workers"
    for f in "$TMP"/fail/*; do [ -e "$f" ] && n=$((n + 1)); done
    printf '%s\n' "$n" > "$TMP/failed"
}

# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------

while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        -o|--output-dir|--local-dir)
            [ $# -ge 2 ] || die "$1 needs a directory"
            OUT_DIR=$2; shift ;;
        -r|--revision)
            [ $# -ge 2 ] || die "$1 needs a revision"
            REVISION=$2; shift ;;
        -t|--type|--repo-type)
            [ $# -ge 2 ] || die "$1 needs model, dataset or space"
            REPO_TYPE=$2; shift ;;
        -j|--jobs)
            [ $# -ge 2 ] || die "$1 needs a number"
            JOBS=$2; shift ;;
        --retries)
            [ $# -ge 2 ] || die "$1 needs a number"
            RETRIES=$2; shift ;;
        --token)
            [ $# -ge 2 ] || die "$1 needs a token"
            TOKEN=$2; shift ;;
        --endpoint)
            [ $# -ge 2 ] || die "$1 needs a URL"
            ENDPOINT=$2; shift ;;
        --include)
            shift
            [ $# -gt 0 ] || die "--include needs at least one glob"
            while [ $# -gt 0 ]; do
                case "$1" in -*) break ;; esac
                INCLUDE+=("$1"); shift
            done
            continue ;;
        --exclude)
            shift
            [ $# -gt 0 ] || die "--exclude needs at least one glob"
            while [ $# -gt 0 ]; do
                case "$1" in -*) break ;; esac
                EXCLUDE+=("$1"); shift
            done
            continue ;;
        --repo)
            [ $# -ge 2 ] || die "--repo needs an id"
            REPO=$2; shift ;;
        --list) LIST_ONLY=1 ;;
        --dry-run) LIST_ONLY=1; DRY_RUN=1 ;;
        --verify-sha) VERIFY_SHA=1 ;;
        --force) FORCE=1 ;;
        --no-disk-check) DISK_CHECK=0 ;;
        --no-progress) PROGRESS=0 ;;
        --no-writer-check) WRITER_CHECK=0 ;;
        --) shift; break ;;
        -*)
            die "unknown option: $1 (try --help)" ;;
        *)
            [ -n "$REPO" ] && die "unexpected argument: $1"
            REPO=$1 ;;
    esac
    shift
done

[ -n "$REPO" ] || { usage >&2; exit 1; }

# Charset check before anything is interpolated into a URL or a path. A repo id
# is [A-Za-z0-9._-] with at most one '/', or the legacy single-segment form.
case "$REPO" in
    *[!A-Za-z0-9._/-]*) die "invalid repo id: $REPO" ;;
    /*|*/|*//*) die "invalid repo id: $REPO" ;;
esac
case "$REVISION" in
    *[!A-Za-z0-9._/-]*) die "invalid revision: $REVISION" ;;
esac
case "$JOBS" in
    ''|*[!0-9]*) die "--jobs must be a number" ;;
esac
[ "$JOBS" -ge 1 ] || die "--jobs must be at least 1"
case "$RETRIES" in
    ''|*[!0-9]*) die "--retries must be a number" ;;
esac

case "$REPO_TYPE" in
    model) RESOLVE_PREFIX="$REPO" ;;
    dataset) RESOLVE_PREFIX="datasets/$REPO" ;;
    space) RESOLVE_PREFIX="spaces/$REPO" ;;
    *) die "--type must be model, dataset or space" ;;
esac

ENDPOINT=${ENDPOINT%/}
case "$ENDPOINT" in
    http://*|https://*) ;;
    *) die "--endpoint must start with http:// or https://" ;;
esac

if [ -z "$TOKEN" ] && [ -s "$HOME/.cache/huggingface/token" ]; then
    TOKEN=$(tr -d '\r\n' < "$HOME/.cache/huggingface/token")
fi

if [ -z "$OUT_DIR" ]; then
    # Keep the org prefix: the layout on disk is <org>/<model>, and dropping it
    # would silently start a second copy of a repository that is already half
    # downloaded under <org>/<model>.
    OUT_DIR="./$REPO"
fi
# A relative -o is anchored to the cwd the caller had, not to wherever the
# script happens to be later.
case "$OUT_DIR" in
    /*) ;;
    ./*) OUT_DIR="$PWD/${OUT_DIR#./}" ;;
    *) OUT_DIR="$PWD/$OUT_DIR" ;;
esac


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

TMP=$(mktemp -d "${TMPDIR:-/tmp}/download_model.XXXXXX") || die "cannot create a temp dir"
mkdir -p "$TMP/active" "$TMP/done"
cleanup() {
    [ -n "${TICKER:-}" ] && kill "$TICKER" 2>/dev/null
    [ -n "${TMP:-}" ] && [ -d "${TMP:-}" ] && rm -rf "$TMP"
    # Only the run that created the lock may remove it: dying because somebody
    # else holds it must not erase their lock on the way out.
    [ "${LOCK_OWNED:-0}" = 1 ] && rm -f "${LOCK:-}"
    return 0
}
on_int() {
    trap - INT TERM
    printf '\ninterrupted; re-run the same command to resume\n' >&2
    [ "$PROG_TTY" = 1 ] && printf '\033[?25h'
    # From here on nothing should be said: bash reports every job it kills
    # ("Terminated: 15 ... curl"), which buries the one line that matters.
    exec 2>/dev/null
    if [ -n "${TMP:-}" ] && [ -d "${TMP:-}" ]; then
        jobs -p > "$TMP/pids" 2>/dev/null || :
        while read -r p; do
            [ -n "$p" ] || continue
            [ "$p" = "${TICKER:-}" ] && continue
            pkill -P "$p" 2>/dev/null || :
            kill "$p" 2>/dev/null || :
        done < "$TMP/pids"
    fi
    cleanup
    exit 130
}
trap on_int INT TERM
trap cleanup EXIT

pick_json_tools

printf 'repo    %s (%s, %s)\n' "$REPO" "$REPO_TYPE" "$REVISION"
printf 'mirror  %s\n' "$ENDPOINT"

list_repo "$TMP/all.tsv"
select_files "$TMP/all.tsv" "$TMP/selected"

count=$(cat "$TMP/count")
[ "$count" -gt 0 ] || die "no files matched the include/exclude filters"

if [ "$LIST_ONLY" = 1 ]; then
    # --list is discovery (everything in the repository), --dry-run is the
    # preview (exactly what this invocation would fetch).
    if [ "$DRY_RUN" = 1 ]; then
        print_table "$TMP/selected"
    else
        print_table "$TMP/all.tsv"
    fi
    exit 0
fi

printf 'files   %s selected\n' "$count"
printf 'into    %s\n' "$OUT_DIR"

mkdir -p "$OUT_DIR" || die "cannot create $OUT_DIR"

# One writer per directory. Two runs (or a run plus huggingface_hub) appending
# to the same .incomplete make the resumed offset stop matching what the server
# believes, which is exactly how a 7.1 MiB file ends up at 7.3 MiB and then
# fails every attempt.
LOCK="$OUT_DIR/.cwick-download.lock"
LOCK_OWNED=0
if [ -e "$LOCK" ]; then
    other=''
    read -r other < "$LOCK" || :
    if [ -n "${other:-}" ] && kill -0 "$other" 2>/dev/null; then
        die "another download (pid $other) is already writing into $OUT_DIR; let it finish, or stop it and re-run"
    fi
    warn "ignoring the lock left by pid ${other:-?} in $OUT_DIR"
fi
# The lock only binds runs that know about it. An older copy of this script, or
# huggingface_hub, writes the same .incomplete files with no lock at all, and
# two appenders turn a resume into garbage - so look for recent write activity
# before claiming the directory.
if [ "$WRITER_CHECK" = 1 ]; then
    busy_list=$(find "$OUT_DIR" -name '*.incomplete' -mmin -1 2>/dev/null)
    if [ -n "$busy_list" ]; then
        # A fresh mtime is not proof on its own: a writer that was killed - the
        # terminal it ran in was closed, the machine slept mid-transfer - leaves
        # exactly the same trace and blocks the resume for a minute for nothing.
        # Ask the kernel who, if anyone, still has the files open. Check EVERY
        # recent file, not just the first one: an interrupted parallel run can
        # leave one orphan curl per shard, and stopping one of them only to
        # resume while the others still append is exactly how a .incomplete
        # becomes garbage.
        holders=''
        held_files=''
        if command -v lsof >/dev/null 2>&1; then
            while IFS= read -r busy; do
                [ -n "$busy" ] || continue
                pids=$(lsof -t "$busy" 2>/dev/null || true)
                for p in $pids; do
                    case " $holders " in
                        *" $p "*) ;;
                        *) holders="$holders $p" ;;
                    esac
                    held_files="$held_files ${busy#$OUT_DIR/}"
                done
            done <<EOF
$busy_list
EOF
            if [ -n "$holders" ]; then
                die "something is already writing into $OUT_DIR (pid(s)${holders} hold${held_files} open). Stop every one of them first; --no-writer-check would corrupt the resume."
            fi
            warn "no process holds the recent .incomplete file(s) open any more; the writer that touched them is gone, resuming from where it stopped"
        else
            die "something changed .incomplete files in $OUT_DIR within the last minute, and lsof is unavailable to check whether a writer is still alive. Stop any downloader first; pass --no-writer-check only if you are certain they are stale."
        fi
    fi
fi
printf '%s\n' "$$" > "$LOCK"
LOCK_OWNED=1

check_ulimit
check_filesystem "$OUT_DIR"
check_disk "$OUT_DIR"

START=$(now)
prog_init
prog_start
if [ "$JOBS" -gt 1 ]; then
    printf 'jobs    %s\n\n' "$JOBS"
    run_parallel
else
    printf '\n'
    run_sequential
fi
prog_stop
ELAPSED=$(( $(now) - START ))

failed=$(cat "$TMP/failed" 2>/dev/null || echo 0)
# What this run moved, not what ended up on disk: a file resumed at 3 MiB of
# 7 MiB fetched 4 MiB here, and saying 7 MiB would inflate the average rate.
bytes=0
if [ -s "$TMP/xfer" ]; then
    bytes=$(awk '{s += $1} END{printf "%d", s}' "$TMP/xfer")
fi

printf '\n'
if [ -f "$TMP/abort" ]; then
    printf 'stopped early: %s\n' "$(cat "$TMP/abort")" >&2
    printf '              re-run once that is fixed; finished files are kept.\n' >&2
fi
if [ "$bytes" -gt 0 ] && [ "$ELAPSED" -gt 0 ]; then
    printf 'fetched %s in %ss (%s/s)\n' \
        "$(human "$bytes")" "$ELAPSED" "$(human $((bytes / ELAPSED)))"
fi
if [ "$failed" -gt 0 ]; then
    printf '%s file(s) failed; re-run the same command to resume them\n' "$failed" >&2
    exit 1
fi
printf 'complete: %s file(s) in %s\n' "$count" "$OUT_DIR"
