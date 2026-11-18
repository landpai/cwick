/* rt_torch.h — read tensors out of a torch.save checkpoint without libtorch.
 *
 * torch.save writes a zip container holding one pickle (data.pkl) that
 * describes the object graph and one raw blob per storage (data/<key>). The
 * blobs are stored uncompressed, so the loader only has to walk the zip
 * central directory and interpret enough of the pickle opcode stream to
 * recover, for each tensor, its storage key, dtype, shape and element offset.
 * Both are done on the mmap of the file: no copy is made for the payload.
 *
 * Used by the speech pipeline (talker LM / flow / vocoder / voice prompts),
 * whose checkpoints are ordinary state dicts rather than safetensors. */
#ifndef CWICK_RT_TORCH_H
#define CWICK_RT_TORCH_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef enum {
    RT_DT_F64 = 0,
    RT_DT_F32,
    RT_DT_F16,
    RT_DT_BF16,
    RT_DT_I64,
    RT_DT_I32,
    RT_DT_I16,
    RT_DT_I8,
    RT_DT_U8,
    RT_DT_BOOL,
    RT_DT_UNKNOWN
} rt_dtype;

typedef struct {
    char name[160];
    rt_dtype dtype;
    int ndim;
    int64_t shape[8];
    int64_t numel;
    const void *data; /* element 0 of a contiguous tensor */
    bool owned;       /* data is a malloc'd copy (non-contiguous source) */
} rt_tensor;

typedef struct rt_pt rt_pt;

/* Opens a checkpoint and indexes every top-level tensor. Returns 0 on
 * success, -1 when the file cannot be mapped or the pickle is not a flat
 * name -> tensor mapping (the callers all use state dicts). */
int rt_pt_open(rt_pt **out, const char *path);
void rt_pt_close(rt_pt *pt);

int rt_pt_count(const rt_pt *pt);
const rt_tensor *rt_pt_at(const rt_pt *pt, int i);
/* Exact-name lookup; returns NULL when absent. */
const rt_tensor *rt_pt_get(const rt_pt *pt, const char *name);
/* Number of tensors whose name starts with prefix. */
int rt_pt_count_prefix(const rt_pt *pt, const char *prefix);

size_t rt_dtype_size(rt_dtype d);
const char *rt_dtype_name(rt_dtype d);
bool rt_tensor_fits(const rt_tensor *t, const int64_t *want_ndim_shape, int want_ndim);

/* Element access in f32 (handles f64/f32/f16/bf16/i64/i32/i16/i8/u8/bool). */
float rt_tensor_f32(const rt_tensor *t, int64_t i);
/* Row access for a 2-D tensor; out must hold shape[1] floats. */
void rt_tensor_row_f32(const rt_tensor *t, int64_t row, float *out);

#endif /* CWICK_RT_TORCH_H */
