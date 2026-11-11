/* models.c — see models.h. */
#include "models.h"

#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "json.h"
#include "log.h"

#define MAX_PATH 4096

void cwick_models_list_init(cwick_model_list *l) {
    l->v = NULL;
    l->n = l->cap = 0;
}

void cwick_models_list_free(cwick_model_list *l) {
    free(l->v);
    l->v = NULL;
    l->n = l->cap = 0;
}

bool cwick_models_list_add(cwick_model_list *l, const char *id, const char *root) {
    if (!id || !*id || !root || !*root) return false;
    for (size_t i = 0; i < l->n; i++) {
        if (strcmp(l->v[i].root, root) == 0) return false;   /* same weights */
    }
    if (l->n == l->cap) {
        size_t cap = l->cap ? l->cap * 2 : 8;
        cwick_model_entry *nv = (cwick_model_entry *)realloc(l->v,
                                                           cap * sizeof(*nv));
        if (!nv) return false;
        l->v = nv;
        l->cap = cap;
    }
    cwick_model_entry *e = &l->v[l->n++];
    memset(e, 0, sizeof(*e));
    snprintf(e->id, sizeof(e->id), "%s", id);
    snprintf(e->root, sizeof(e->root), "%s", root);
    if (cwick_model_served_alias(root, e->alias, sizeof(e->alias)) &&
        strcmp(e->alias, e->id) == 0)
        e->alias[0] = '\0';   /* no better name than what it calls itself */
    /* One small config read at discovery time, not on every catalog poll. */
    char why[192];
    e->supported = cwick_model_root_supported(root, why, sizeof(why));
    if (!e->supported)
        snprintf(e->unsupported_reason, sizeof(e->unsupported_reason), "%s", why);
    return true;
}

/* The name a checkpoint answers to. `model_type` inside config.json is a
 * training-run artifact ("glm5_next") that no client would guess; the release
 * checkpoints get their familiar names spelled out here, keyed by directory
 * basename. Everything else keeps its own model_type — inventing a name for
 * an arbitrary local checkpoint (a fixture, a fine-tune) would churn every
 * client that already learned the old one for no gain. */
bool cwick_model_served_alias(const char *root, char *out, size_t cap) {
    static const struct {
        const char *base;
        const char *served;
    } known[] = {
        {"GLM-5.3-Flash", "glm-5.3-flash"},
        {"DeepSeek-V4.1-Flash-1017", "deepSeek-v4.1-flash-1017"},
        {"MiniMax-M3.1-Flash", "MiniMax-M3.1-Flash"},
    };
    if (!root || !*root || !out || cap == 0) return false;
    out[0] = '\0';
    const char *slash = strrchr(root, '/');
    const char *base = slash ? slash + 1 : root;
    if (*base == '\0') base = root;
    for (size_t i = 0; i < sizeof(known) / sizeof(known[0]); i++) {
        if (strcmp(base, known[i].base) == 0) {
            snprintf(out, cap, "%s", known[i].served);
            return true;
        }
    }
    return false;
}

void cwick_models_list_dedupe_ids(cwick_model_list *l) {
    /* Two checkpoints can legitimately report the same model_type (two fine
     * tunes of one base, two copies of the fixture tree). /v1/models must
     * still be able to say which one it means, so give the later arrivals a
     * numeric suffix — and keep searching suffixes until the new id collides
     * with nothing, because renaming as we go changes what later comparisons
     * see.
     *
     * Entry 0 is never touched: it is the resident model, and its id is the
     * one this instance announces at startup and validates requests against.
     * Renaming it would silently break every client that learned the name
     * from that banner. */
    for (size_t i = 1; i < l->n; i++) {
        bool clash = false;
        for (size_t j = 0; j < l->n; j++) {
            if (j != i && strcmp(l->v[j].id, l->v[i].id) == 0) { clash = true; break; }
        }
        if (!clash) continue;

        char base[CWICK_MODEL_ID_MAX];
        snprintf(base, sizeof(base), "%s", l->v[i].id);
        for (int k = 2; k < 1000; k++) {
            char cand[CWICK_MODEL_ID_MAX];
            snprintf(cand, sizeof(cand), "%s@%d", base, k);
            bool taken = false;
            for (size_t j = 0; j < l->n; j++) {
                if (j != i && strcmp(l->v[j].id, cand) == 0) { taken = true; break; }
            }
            if (!taken) {
                snprintf(l->v[i].id, sizeof(l->v[i].id), "%s", cand);
                break;
            }
        }
    }

    /* The same argument applies to the served alias: two local checkpoints
     * with the same directory basename (a release copy and a working tree)
     * would otherwise advertise one OpenAI name for two roots. Same scheme,
     * same carve-out for entry 0. */
    for (size_t i = 1; i < l->n; i++) {
        if (!l->v[i].alias[0]) continue;
        bool clash = false;
        for (size_t j = 0; j < l->n; j++) {
            if (j == i) continue;
            if (strcmp(l->v[j].alias, l->v[i].alias) == 0 ||
                strcmp(l->v[j].id, l->v[i].alias) == 0) { clash = true; break; }
        }
        if (!clash) continue;

        char base[CWICK_MODEL_ID_MAX];
        snprintf(base, sizeof(base), "%s", l->v[i].alias);
        for (int k = 2; k < 1000; k++) {
            char cand[CWICK_MODEL_ID_MAX];
            snprintf(cand, sizeof(cand), "%s@%d", base, k);
            bool taken = false;
            for (size_t j = 0; j < l->n; j++) {
                if (j == i) continue;
                if (strcmp(l->v[j].alias, cand) == 0 ||
                    strcmp(l->v[j].id, cand) == 0) { taken = true; break; }
            }
            if (!taken) {
                snprintf(l->v[i].alias, sizeof(l->v[i].alias), "%s", cand);
                break;
            }
        }
    }
}

/* ---- probing ------------------------------------------------------ */

static bool file_readable(const char *path) {
    struct stat st;
    return stat(path, &st) == 0 && S_ISREG(st.st_mode) &&
           access(path, R_OK) == 0;
}
static bool dir_readable(const char *path) {
    struct stat st;
    return stat(path, &st) == 0 && S_ISDIR(st.st_mode) &&
           access(path, R_OK) == 0;
}

/* Read a small text file whole. config.json is tens of bytes; a pathological
 * hundreds-of-megabytes one is refused rather than mapped into memory. */
static char *read_small_file(const char *path, size_t limit) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    char *buf = (char *)malloc(limit + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t n = fread(buf, 1, limit, f);
    bool overflow = false;
    if (n == limit && !feof(f)) overflow = true;
    fclose(f);
    if (overflow) { free(buf); return NULL; }
    buf[n] = '\0';
    return buf;
}

static void basename_of(const char *path, char *out, size_t cap) {
    const char *slash = strrchr(path, '/');
    const char *name = slash ? slash + 1 : path;
    if (*name == '\0') name = path;
    snprintf(out, cap, "%s", name);
}

bool cwick_model_dir_probe(const char *dir, char *id_out, size_t id_cap) {
    if (!dir || !*dir) return false;

    char cfg[MAX_PATH];
    if (snprintf(cfg, sizeof(cfg), "%s/config.json", dir) >= (int)sizeof(cfg))
        return false;
    if (!file_readable(cfg)) return false;

    /* Weights: the converted tree, or the released HF shards. */
    char probe[MAX_PATH];
    bool has_weights = false;
    if (snprintf(probe, sizeof(probe), "%s/resident/tensors", dir) <
            (int)sizeof(probe) &&
        dir_readable(probe))
        has_weights = true;
    if (!has_weights &&
        snprintf(probe, sizeof(probe), "%s/model.safetensors.index.json",
                 dir) < (int)sizeof(probe) &&
        file_readable(probe))
        has_weights = true;
    if (!has_weights &&
        snprintf(probe, sizeof(probe), "%s/model.safetensors", dir) <
            (int)sizeof(probe) &&
        file_readable(probe))
        has_weights = true;
    if (!has_weights) return false;

    if (id_out && id_cap) {
        basename_of(dir, id_out, id_cap);
        char *text = read_small_file(cfg, 1u << 20);
        if (text) {
            cwick_error err;
            cwick_error_clear(&err);
            json_value *root = cwick_json_parse(text, strlen(text), &err);
            if (root) {
                const char *mt = cwick_json_str(cwick_json_get(root, "model_type"));
                if (mt && *mt) {
                    snprintf(id_out, id_cap, "%s", mt);
                } else {
                    json_value *ar = cwick_json_get(root, "architectures");
                    if (ar && ar->kind == JSON_ARRAY && ar->count > 0) {
                        const char *a = cwick_json_str(ar->items[0]);
                        if (a && *a) snprintf(id_out, id_cap, "%s", a);
                    }
                }
                cwick_json_free(root);
            }
            free(text);
        }
    }
    return true;
}

/* ------------------------------------------------------------------ */
/* Architecture compatibility                                          */
/* ------------------------------------------------------------------ */
#define TYPE_HAS_PREFIX(mt, lit) (strncmp((mt), (lit), sizeof(lit) - 1) == 0)

bool cwick_model_type_supported(const char *mt) {
    if (!mt || !*mt) return false;
    if (strcmp(mt, "deepseek_v41") == 0 ||
        strcmp(mt, "deepseek_v41_text") == 0) return true;
    if (TYPE_HAS_PREFIX(mt, "minimax_m3")) return true;
    if (strcmp(mt, "glm5_next") == 0 ||
        strcmp(mt, "glm-5.3-flash") == 0) return true;
    /* `cwick setup` writes tiny checkpoints that use the same kernels. */
    if (TYPE_HAS_PREFIX(mt, "glm_5_3_flash_synthetic") ||
        TYPE_HAS_PREFIX(mt, "glm-5.3-flash-synthetic")) return true;
    return false;
}

bool cwick_model_root_supported(const char *root, char *why, size_t whycap) {
    if (why && whycap) why[0] = '\0';
    if (!root || !*root) {
        if (why && whycap) snprintf(why, whycap, "empty model root");
        return false;
    }
    char cfg[MAX_PATH];
    if (snprintf(cfg, sizeof(cfg), "%s/config.json", root) >= (int)sizeof(cfg)) {
        if (why && whycap) snprintf(why, whycap, "model root is too long");
        return false;
    }
    char *text = read_small_file(cfg, 1u << 20);
    if (!text) {
        if (why && whycap) snprintf(why, whycap, "cannot read %s", cfg);
        return false;
    }
    cwick_error err;
    cwick_error_clear(&err);
    json_value *j = cwick_json_parse(text, strlen(text), &err);
    free(text);
    if (!j) {
        if (why && whycap) snprintf(why, whycap, "%s is not valid JSON", cfg);
        return false;
    }
    const char *outer = cwick_json_str(cwick_json_get(j, "model_type"));
    const char *inner = NULL;
    json_value *tc = cwick_json_get(j, "text_config");
    if (tc && tc->kind == JSON_OBJECT)
        inner = cwick_json_str(cwick_json_get(tc, "model_type"));
    /* The loader unwraps `text_config` first and only then falls back to the
     * outer object; the compatibility answer must use the same precedence or
     * a nested config would be advertised as supported and then refused. */
    const char *mt = inner ? inner : outer;
    bool ok = cwick_model_type_supported(mt);
    if (!ok && why && whycap)
        snprintf(why, whycap,
                 "unsupported model_type \"%s\": this build implements the "
                 "GLM-5.3 / DeepSeek-V4.1 / MiniMax-M3.1 kernels only",
                 mt ? mt : "(missing)");
    cwick_json_free(j);
    return ok;
}
#undef TYPE_HAS_PREFIX

int cwick_models_scan_dir(const char *dir, cwick_model_list *l) {
    if (!dir || !*dir) return 0;
    struct stat st;
    if (stat(dir, &st) != 0) {
        CWICK_INFO("models", "--models-dir %s is not readable (skipped)", dir);
        return -1;
    }
    /* A directory that *is* a checkpoint: list it as itself. */
    int added = 0;
    char id[CWICK_MODEL_ID_MAX];
    if (cwick_model_dir_probe(dir, id, sizeof(id)) &&
        cwick_models_list_add(l, id, dir))
        added++;

    DIR *d = opendir(dir);
    if (!d) return added;                 /* a file, not a directory: fine */

    struct dirent *de;
    while ((de = readdir(d)) != NULL) {
        if (strcmp(de->d_name, ".") == 0 || strcmp(de->d_name, "..") == 0)
            continue;
        if (de->d_name[0] == '.') continue;      /* .cache, .DS_Store, … */
        char child[MAX_PATH];
        if (snprintf(child, sizeof(child), "%s/%s", dir, de->d_name) >=
            (int)sizeof(child))
            continue;
        char cid[CWICK_MODEL_ID_MAX];
        if (!cwick_model_dir_probe(child, cid, sizeof(cid))) continue;
        if (cwick_models_list_add(l, cid, child)) added++;
    }
    closedir(d);
    return added;
}
