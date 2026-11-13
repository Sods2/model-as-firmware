// PC build of the MafLM runtime, for checking it against train/lm/intref.py before it goes on the board.
//   host_main <tokens.bin> <logits.bin>
// Runs lm_step() over every byte of tokens.bin from a fresh KV cache and writes the int64 logits of every
// position (n x LM_VOCAB, little-endian) to logits.bin.
#include <stdio.h>
#include <stdlib.h>
#include "lm_rt.h"

int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr, "usage: host_main tokens.bin logits.bin\n"); return 2; }
    FILE *f = fopen(argv[1], "rb"), *g = fopen(argv[2], "wb");
    if (!f || !g) { perror("open"); return 2; }
    lm_init();
    int c, n = 0;
    while ((c = fgetc(f)) != EOF) {
        lm_step(c);
        fwrite(lm_logits, sizeof lm_logits[0], LM_VOCAB, g);
        n++;
    }
    fclose(f);
    fclose(g);
    printf("%s: %d positions\n", LM_MODEL_NAME, n);
    return 0;
}
