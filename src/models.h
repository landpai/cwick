/* models.h — discover locally available cwick checkpoints.
 *
 * `cwick serve --root DIR` boots exactly one model, and that is the right
 * default for a 306 GiB checkpoint: the resident spine alone is most of the
 * machine's 80% budget, so "resident everything" is not on offer. But a GUI
 * that can only ever talk to whatever was passed on the command line is a GUI
 * you restart by hand every time you want the other model, and a second model
 * that fits (a smaller fine-tune, a bf16 conversion) costs nothing to know
 * about.
 *
 * So: scan a set of directories (repeatable `--models-dir`), remember what
 * each root is and what it calls itself, advertise all of them through
 * /v1/models, and let POST /v1/models/switch swap the resident one. Switching
 * costs a reload, which for the real checkpoint is minutes — it is the honest
 * price of "no pruning, full weights, 128 GB".
 *
 * Discovery is deliberately shallow (each root's own directory plus its
 * immediate children) and read-only: no recursion into a checkpoint's shard
 * trees, no writes anywhere. */
#ifndef CWICK_MODELS_H
#define CWICK_MODELS_H

#include <stdbool.h>
#include <stddef.h>

#define CWICK_MODEL_ID_MAX 128
#define CWICK_MODEL_ROOT_MAX 1024

typedef struct {
    char id[CWICK_MODEL_ID_MAX];
    /* The name this checkpoint answers to in OpenAI payloads. `id` is whatever
     * the checkpoint's config.json calls itself (`model_type`), which is a
     * training-run artifact (`glm5_next`) and means nothing to a client; the
     * alias is derived from the checkpoint directory and is what /v1/models
     * advertises and what completions echo back. The checkpoint id stays
     * accepted everywhere so nothing that learned it breaks. Empty when no
     * better name exists. */
    char alias[CWICK_MODEL_ID_MAX];
    char root[CWICK_MODEL_ROOT_MAX];
} cwick_model_entry;

typedef struct {
    cwick_model_entry *v;
    size_t n, cap;
} cwick_model_list;

void cwick_models_list_init(cwick_model_list *l);
void cwick_models_list_free(cwick_model_list *l);
/* Append unless an entry with this exact root is already listed. Returns true
 * when something was added. */
bool cwick_models_list_add(cwick_model_list *l, const char *id, const char *root);
/* In-place "is this id unique" helper: rewrite colliding ids as `id@2`,
 * because /v1/models advertising two models with the same id would let a
 * client ask for one and get the other. */
void cwick_models_list_dedupe_ids(cwick_model_list *l);

/* The served name for a checkpoint at `root`, looked up by directory
 * basename: a release checkpoint gets its familiar short name ("GLM-5.3-Flash"
 * -> "glm-5.3-flash"). Returns false (and leaves `out` empty) for anything
 * else, meaning the checkpoint's own model_type should be used as-is. */
bool cwick_model_served_alias(const char *root, char *out, size_t cap);

/* Does `dir` look like a cwick checkpoint root? True when `config.json` sits
 * next to a converted `resident/tensors` tree or the released HF shards
 * (`model.safetensors.index.json` / `model.safetensors`). On success writes
 * the name the checkpoint calls itself into `id_out` (config.json's
 * `model_type`, else `architectures[0]`, else the directory name). */
bool cwick_model_dir_probe(const char *dir, char *id_out, size_t id_cap);

/* Scan `dir` and each immediate subdirectory, appending every model found.
 * Returns how many were added, or -1 if `dir` cannot be read at all (a
 * misconfiguration worth shouting about, not worth failing startup over). */
int cwick_models_scan_dir(const char *dir, cwick_model_list *l);

#endif /* CWICK_MODELS_H */
