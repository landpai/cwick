/* q35_gpu.h — one-position Metal forward for the Qwen3.5 checkpoint.
 *
 * Optional and best-effort: q35_model_forward falls back to the CPU path when
 * q35_gpu_step() declines, and q35_gpu_close() is safe whether or not the
 * backend ever came up. See q35_gpu.c for what it is for and why it is shaped
 * the way it is.
 */
#ifndef CWICK_Q35_GPU_H
#define CWICK_Q35_GPU_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct q35_model;

/* Verified positions per step. A speculative step forwards k consecutive
 * positions through one weight pass, so this is the widest verify batch the
 * scratch buffers are sized for. */
#define Q35_GK 8

/* Copies of the recurrent state the model holds, one per row of a verify plus
 * the live one. A k-row step runs row r into slot r+1 from slot r, so the
 * state that belongs to "the first j rows were kept" is still there afterwards
 * -- rewinding a DeltaNet is not possible, and re-running the rows to rebuild
 * it would cost a whole weight pass. Slot 0 is always the live state: the CPU
 * path, the prefix cache and the k-row step's first row all use it. */
#define Q35_STATE_SLOTS (Q35_GK + 1)

/* Should the GPU take decode steps for this process? Reads CWICK_Q35_GPU and
 * requires a working Metal device. */
int q35_gpu_available(void);

/* Bind weights and caches, allocate device scratch. 0 on failure, in which
 * case the model keeps the CPU path and nothing else changes. */
int q35_gpu_open(struct q35_model *m);
void q35_gpu_close(void);

/* One decode step for `id` at absolute position `pos`. Returns 1 when the GPU
 * produced `logits`, 0 when the caller must run the CPU path instead. */
int q35_gpu_step(struct q35_model *m, int32_t id, int64_t pos, float *logits);

/* `n` consecutive positions from `ids`, one weight pass, logits written to
 * logits[0 .. n * vocab_size). The rows are verified in order, so the KV
 * cache, the DeltaNet state and the convolution window advance exactly as n
 * single steps would -- the only thing shared is the weight read. Returns 1 on
 * success, 0 when the GPU declines the shape, -1 on a device failure. */
int q35_gpu_step_n(struct q35_model *m, const int32_t *ids, int64_t pos,
                   int n, float *logits);

/* Can a k-row step be encoded at all (a pipeline state for this k)? */
int q35_gpu_supports_n(int n);

/* After a k-row step, keep the state as of row `j` (0 <= j < k): row j's copy
 * becomes the live slot 0, and every slot above it is left for the next pass
 * to overwrite. Returns 1 on success, 0 when there was no such slot. */
int q35_gpu_commit_rows(int j);

#ifdef __cplusplus
}
#endif

#endif /* CWICK_Q35_GPU_H */
