// PC build of the reference, to check the C code before it goes on the board.
// Usage: ref_host <images.bin>
//   images.bin: N x 256 int8 images, back to back.
// Prints one line per image in the same format as the Pico firmware:
//   R <pred> <cycles=0> <logit0> ... <logit9>
#include <stdio.h>
#include <stdint.h>
#include "ref_infer.h"

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <images.bin>\n", argv[0]);
        return 2;
    }
    FILE *f = fopen(argv[1], "rb");
    if (!f) {
        perror(argv[1]);
        return 1;
    }
    int8_t img[REF_IMG_PIXELS];
    int32_t logits[REF_N_CLASSES];
    while (fread(img, 1, sizeof img, f) == sizeof img) {
        uint32_t pred = ref_infer(img, logits);
        printf("R %u 0", (unsigned)pred);
        for (int i = 0; i < REF_N_CLASSES; i++) printf(" %ld", (long)logits[i]);
        printf("\n");
    }
    fclose(f);
    return 0;
}
