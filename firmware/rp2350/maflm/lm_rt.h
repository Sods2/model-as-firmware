// MafLM integer runtime: one generated token at a time, with a KV cache.
// Bit-exact with train/lm/intref_step.py (and so with the batched reference train/lm/intref.py); the arithmetic is
// defined in intref.py's docstring. Model data and LM_* constants come from the compiler (gen/maflm_<model>.h/.c).
#pragma once
#include <stdint.h>
#include LM_GEN_HEADER

// A ternary linear layer: rows x cols trits, one byte per 4 trits holding the base-3 number d0 + 3 d1 + 9 d2 + 27 d3
// (digit 0 = 0, 1 = +1, 2 = -1); cols is a multiple of 4. wmul = round(wscale * 2^(F+16) / 127).
// Layout il = 0: row-major bytes. il = 1 (rows a multiple of 4): rows interleaved in groups of 4, so that byte r of
// 32-bit word (g * cols/4 + i) is input group i of row 4g + r: one load fetches four rows' weights.
typedef struct {
    const uint8_t *w;
    uint16_t rows, cols;
    int32_t wmul;
    uint8_t il;
} lm_lin_t;

typedef struct {
    lm_lin_t wq, wk, wv, wo;
#if LM_TREE
    lm_lin_t nodes, leaf_in, leaf_out[1 << LM_DEPTH];
#else
    lm_lin_t w1, w2;
#endif
} lm_block_t;

typedef struct {
    const int8_t *embed_codes;    // [LM_VOCAB][LM_D]
    const int32_t *embed_mul;     // [LM_VOCAB]
    lm_block_t blocks[LM_LAYERS];
    lm_lin_t unembed;
} lm_model_t;

extern const lm_model_t lm_model;

// Stage cycle counters (LM_PROFILE builds): q/k/v projections, attention core, Wo, FFN first half (quantize x,
// W1 or nodes + leaf_in), FFN second half (ReLU, quantize, W2 or leaf_out), unembed. LM_ST_TAB counts the
// activation-table builds, which are also inside the other stages.
enum { LM_ST_QKV, LM_ST_ATTN, LM_ST_WO, LM_ST_FFN1, LM_ST_FFN2, LM_ST_UNEMBED, LM_ST_N, LM_ST_TAB = LM_ST_N };
extern uint32_t lm_stage_cycles[LM_ST_N + 1];

void lm_init(void);                          // builds the trit lookup table; call once
void lm_reset(void);                         // empties the KV cache (start of a new text)
// Runs one token through the model at the next position. Returns the greedy next token (first maximum);
// the logits (residual units, 2^-LM_F) are in lm_logits.
int lm_step(int token);
extern int64_t lm_logits[LM_VOCAB];
extern int lm_pos;                           // tokens processed since lm_reset()
