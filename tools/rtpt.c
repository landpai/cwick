/* rtpt.c — dump a torch.save checkpoint's tensor index.
 *
 *   tools/rtpt <file.pt> [name-filter]
 *
 * Prints one line per tensor: name, dtype, shape, element count, and the
 * first/last element as f32 so a numeric sanity check is one eyeball away.
 * Used to verify src/rt_torch.c against an independent listing of the same
 * checkpoint, and later to confirm loader shapes inside the speech pipeline. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rt_torch.h"

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <file.pt> [filter]\n", argv[0]);
        return 2;
    }
    const char *filter = argc > 2 ? argv[2] : NULL;
    rt_pt *pt = NULL;
    if (rt_pt_open(&pt, argv[1]) != 0) {
        fprintf(stderr, "open failed: %s\n", argv[1]);
        return 1;
    }
    printf("# %s: %d tensors\n", argv[1], rt_pt_count(pt));
    for (int i = 0; i < rt_pt_count(pt); i++) {
        const rt_tensor *t = rt_pt_at(pt, i);
        if (filter && !strstr(t->name, filter)) continue;
        printf("%-64s %-5s [", t->name, rt_dtype_name(t->dtype));
        for (int d = 0; d < t->ndim; d++) printf("%s%lld", d ? "," : "", (long long)t->shape[d]);
        printf("] n=%lld", (long long)t->numel);
        if (t->numel > 0)
            printf(" first=%.6g last=%.6g", rt_tensor_f32(t, 0), rt_tensor_f32(t, t->numel - 1));
        printf("\n");
    }
    rt_pt_close(pt);
    return 0;
}
