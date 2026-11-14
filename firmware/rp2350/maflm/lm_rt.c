// MafLM integer runtime (v0). See lm_rt.h; arithmetic defined in train/lm/intref.py, mirrored step by step from
// train/lm/intref_step.py. Every rounding and shift here must match it exactly.
#include <string.h>
#include "lm_rt.h"

#if defined(__ARM_FEATURE_DSP)
#include <arm_acle.h>
#define SMLAD(a, b, acc) __smlad((a), (b), (acc))
#define SXTB16(x) __sxtb16(x)
#define ROR8(x) __ror((x), 8)
#define SADD16(a, b) ((uint32_t)__sadd16((a), (b)))
#else   // PC build: same results in plain C
static inline uint32_t SADD16(uint32_t a, uint32_t b) {
    return (uint16_t)((int16_t)(a & 0xffff) + (int16_t)(b & 0xffff)) |
           ((uint32_t)(uint16_t)((int16_t)(a >> 16) + (int16_t)(b >> 16)) << 16);
}
static inline uint32_t SXTB16(uint32_t x) {
    return (uint16_t)(int16_t)(int8_t)(x & 0xff) | ((uint32_t)(uint16_t)(int16_t)(int8_t)((x >> 16) & 0xff) << 16);
}
#define ROR8(x) (((x) >> 8) | ((x) << 24))
static inline int32_t SMLAD(uint32_t a, uint32_t b, int32_t acc) {
    return acc + (int16_t)(a & 0xffff) * (int16_t)(b & 0xffff) + (int16_t)(a >> 16) * (int16_t)(b >> 16);
}
#endif

#if defined(PICO_RP2350)
#include "pico.h"
#define HOT __not_in_flash_func
#define DWT_CYCCNT (*(volatile uint32_t *)0xE0001004u)
#else
#define HOT(f) f
#define DWT_CYCCNT 0u
#endif

#ifdef LM_PROFILE
#define PROF_T0() uint32_t _t = DWT_CYCCNT
#define PROF(st) do { uint32_t _n = DWT_CYCCNT; lm_stage_cycles[st] += _n - _t; _t = _n; } while (0)
#else
#define PROF_T0() (void)0
#define PROF(st) (void)0
#endif

// ------------------------------------------------------------------------------------------ two cores (v2)
// par(f, ctx, n, split): core 0 runs f(ctx, 0, split) while core 1 runs f(ctx, split, n); returns when both are done.
// Jobs write disjoint outputs. Core 1 waits on the SIO FIFO (one word each way per job, __dmb() around the shared
// job descriptor). The PC build of a dual-core configuration runs both halves on one core, so the PC check covers
// the same partitioning.
typedef void (*job_fn)(const void *ctx, int lo, int hi);
#if LM_DUAL && defined(PICO_RP2350)
#include "pico/multicore.h"
#include "hardware/sync.h"
static struct { job_fn f; const void *ctx; int lo, hi; } volatile core1_job;
static void __not_in_flash_func(core1_main)(void) {
    for (;;) {
        (void)multicore_fifo_pop_blocking();
        __dmb();
        core1_job.f(core1_job.ctx, core1_job.lo, core1_job.hi);
        __dmb();
        multicore_fifo_push_blocking(1);
    }
}
static void core1_start(void) { multicore_launch_core1(core1_main); }
static inline void par(job_fn f, const void *ctx, int n, int split) {
    if (split <= 0 || split >= n) { f(ctx, 0, n); return; }
    core1_job.f = f; core1_job.ctx = ctx; core1_job.lo = split; core1_job.hi = n;
    __dmb();
    multicore_fifo_push_blocking(1);
    f(ctx, 0, split);
    (void)multicore_fifo_pop_blocking();
    __dmb();
}
#else
static void core1_start(void) {}
static inline void par(job_fn f, const void *ctx, int n, int split) {
#if LM_DUAL
    if (split <= 0 || split >= n) { f(ctx, 0, n); return; }
    f(ctx, split, n);
    f(ctx, 0, split);
#else
    (void)split;
    f(ctx, 0, n);
#endif
}
#endif

#define LM_MAXN 1024
#define LM_KVD (LM_KV_HEADS * LM_D_HEAD)

uint32_t lm_stage_cycles[LM_ST_N + 1];
int64_t lm_logits[LM_VOCAB];
int lm_pos;

static const int32_t EXP2[256] = {
#include "lm_exp2.inc"
};

// ---------------------------------------------------------------------------------------------- KV cache
static int8_t kc[LM_LAYERS][LM_WINDOW][LM_KVD] __attribute__((aligned(4))), vc[LM_LAYERS][LM_WINDOW][LM_KVD];
static int32_t km[LM_LAYERS][LM_WINDOW][LM_KV_HEADS], vm[LM_LAYERS][LM_WINDOW][LM_KV_HEADS];

void lm_init(void) {
    core1_start();
    lm_reset();
}

void lm_reset(void) {
    lm_pos = 0;
    memset(lm_stage_cycles, 0, sizeof lm_stage_cycles);
}

// ----------------------------------------------------------------------------------------- quantized input
typedef struct {
    int8_t c[LM_MAXN] __attribute__((aligned(4)));   // int8 codes
    int64_t m;                                  // max |xs|
    int64_t r;                                  // max(1, isqrt((sum xs^2 / n) << 14))
    int n;
    uint32_t version;                           // bumped by quant(), so cached tables know when they are stale
} qvec_t;

static uint64_t isqrt64(uint64_t v) {
    uint64_t r = 0, bit = 1ull << 62;
    while (bit > v) bit >>= 2;
    while (bit) {
        if (v >= r + bit) { v -= r + bit; r = (r >> 1) + bit; } else r >>= 1;
        bit >>= 2;
    }
    return r;
}

static inline int bitlen64(uint64_t v) { return v ? 64 - __builtin_clzll(v) : 0; }

static inline int64_t code_of(int64_t x, int64_t m) {   // sign(x) * ((254|x| + m) / 2m), m > 0
    int64_t a = x < 0 ? -x : x, q = (254 * a + m) / (2 * m);
    return x < 0 ? -q : q;
}

// v1: after the shift every |xs| <= 2^15, so the code division fits 32 bits (254 * 2^15 + m < 2^24): the M33's
// hardware UDIV instead of a 64-bit library division per element.
static inline int32_t code32(int32_t x, uint32_t m, uint32_t m2) {    // sign(x) * ((254|x| + m) / 2m)
    uint32_t a = x < 0 ? (uint32_t)-x : (uint32_t)x, q = (254 * a + m) / m2;
    return x < 0 ? -(int32_t)q : (int32_t)q;
}

static uint32_t qversion;

static void HOT(quant)(const int64_t *x, int n, qvec_t *q) {
    int64_t mx = 0;
    for (int i = 0; i < n; i++) { int64_t a = x[i] < 0 ? -x[i] : x[i]; if (a > mx) mx = a; }
    int sh = bitlen64((uint64_t)mx) - 15;
    if (sh < 0) sh = 0;
    static int32_t xs[LM_MAXN];
    uint32_t m = 0;
    uint64_t s2 = 0;
    for (int i = 0; i < n; i++) {
        int32_t v = (int32_t)(x[i] >> sh);
        uint32_t a = v < 0 ? (uint32_t)-v : (uint32_t)v;
        xs[i] = v;
        if (a > m) m = a;
        s2 += a * a;
    }
    q->n = n;
    q->m = m;
    q->version = ++qversion;
    int64_t r = (int64_t)isqrt64((s2 / (uint32_t)n) << 14);
    q->r = r < 1 ? 1 : r;
    if (m == 0) { memset(q->c, 0, n); return; }
    const uint32_t m2 = 2 * m;
    for (int i = 0; i < n; i++) q->c[i] = (int8_t)code32(xs[i], m, m2);
}

// ------------------------------------------------------------------------- ternary matvec: activation tables
// v1 kernel (idea E, lookup tables over activations). Weights are stored as one byte per 4 trits, the base-3 number
// d0 + 3 d1 + 9 d2 + 27 d3 (digit 0 = 0, 1 = +1, 2 = -1). For each group of 4 inputs, tab[g][0..80] holds the
// partial sum of every possible trit pattern, so a row's dot product is one table load and one add per weight byte.
// Tables are built LM_TCHUNK inputs at a time (|entry| <= 4 * 128, int16) and reused by every matvec that reads the
// same quantized vector (q, k, v; the tree's nodes and leaf), tagged by the vector's version.
#define PK16(lo, hi) ((uint32_t)(uint16_t)(int16_t)(lo) | ((uint32_t)(uint16_t)(int16_t)(hi) << 16))
typedef uint32_t u32_unaligned __attribute__((aligned(2), may_alias));
#define LM_TCHUNK 128
#define LM_TG (LM_TCHUNK / 4)
static int16_t tab[LM_TG][81];
static uint32_t tab_version, tab_chunk;

typedef struct { const int8_t *c; } tab_job_t;

static void HOT(tab_groups)(const void *ctx, int lo, int hi) {
    const int8_t *c = ((const tab_job_t *)ctx)->c + 4 * lo;
    for (int g = lo; g < hi; g++, c += 4) {
        // T[a + 9b] = A[a] + B[b], with A[i + 3j] = v_i(x0) + v_j(x1) and B[k + 3l] = v_k(x2) + v_l(x3), where
        // v_0(x) = 0, v_1(x) = x, v_2(x) = -x. Each row of 9 entries is 4 SADD16 + 4 word stores (unaligned
        // stores are fine on the M33) + 1 halfword.
        int16_t *t = tab[g];
        const int32_t x0 = c[0], x1 = c[1], x2 = c[2], x3 = c[3];
        const int32_t A[9] = { 0, x0, -x0, x1, x0 + x1, x1 - x0, -x1, x0 - x1, -x0 - x1 };
        const int32_t B[9] = { 0, x2, -x2, x3, x2 + x3, x3 - x2, -x3, x2 - x3, -x2 - x3 };
        const uint32_t a01 = PK16(A[0], A[1]), a23 = PK16(A[2], A[3]), a45 = PK16(A[4], A[5]), a67 = PK16(A[6], A[7]);
        for (int b = 0; b < 9; b++, t += 9) {
            const uint32_t bb = PK16(B[b], B[b]);
            u32_unaligned *tw = (u32_unaligned *)t;   // no STRD/STM merging on a 2-byte-aligned address
            tw[0] = SADD16(a01, bb); tw[1] = SADD16(a23, bb); tw[2] = SADD16(a45, bb); tw[3] = SADD16(a67, bb);
            t[8] = (int16_t)(A[8] + B[b]);
        }
    }
}

static void HOT(build_tab)(const qvec_t *q, int chunk) {
    if (tab_version == q->version && tab_chunk == (uint32_t)chunk) return;
    PROF_T0();
    int groups = (q->n - chunk * LM_TCHUNK) / 4;
    if (groups > LM_TG) groups = LM_TG;
    const tab_job_t job = { q->c + chunk * LM_TCHUNK };
    par(tab_groups, &job, groups, groups / 2);
    tab_version = q->version;
    tab_chunk = (uint32_t)chunk;
    PROF(LM_ST_TAB);
}

// acc[r] = W[row0 + r] . q for r < nrows. Interleaved layout (il): row0 and nrows multiples of 4, one 32-bit load
// per four rows' weight bytes; the row groups are split across the cores. Row-major layout: one row at a time on
// the calling core (the tree nodes). If y is given, the last chunk also writes y = (acc * G + 2^15) >> 16.
typedef struct {
    const uint32_t *w;      // first weight word of row group 0, for this chunk
    int cb, ng, first;
    int32_t *acc;
    int64_t *y, G;
} rows_job_t;

static void HOT(rows_il)(const void *ctx, int lo, int hi) {
    const rows_job_t *J = ctx;
    const uint32_t *w = J->w + (uint32_t)lo * J->cb;
    for (int rg = lo; rg < hi; rg++, w += J->cb) {
        int32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        const int16_t *t = tab[0];
        const uint32_t *wp = w, *we = w + J->ng;
        while (wp + 1 < we) {                       // two groups per iteration, to hide load latency
            const uint32_t w4 = wp[0], v4 = wp[1];
            const int16_t *u = t + 81;
            a0 += t[w4 & 255];          a1 += t[(w4 >> 8) & 255];
            a2 += t[(w4 >> 16) & 255];  a3 += t[w4 >> 24];
            a0 += u[v4 & 255];          a1 += u[(v4 >> 8) & 255];
            a2 += u[(v4 >> 16) & 255];  a3 += u[v4 >> 24];
            wp += 2;
            t += 162;
        }
        if (wp < we) {
            const uint32_t w4 = *wp;
            a0 += t[w4 & 255]; a1 += t[(w4 >> 8) & 255]; a2 += t[(w4 >> 16) & 255]; a3 += t[w4 >> 24];
        }
        int32_t *acc = J->acc + 4 * rg;
        if (!J->first) { a0 += acc[0]; a1 += acc[1]; a2 += acc[2]; a3 += acc[3]; }
        acc[0] = a0; acc[1] = a1; acc[2] = a2; acc[3] = a3;
        if (J->y) {
            int64_t *y = J->y + 4 * rg;
            const int64_t G = J->G;
            y[0] = ((int64_t)a0 * G + (1 << 15)) >> 16; y[1] = ((int64_t)a1 * G + (1 << 15)) >> 16;
            y[2] = ((int64_t)a2 * G + (1 << 15)) >> 16; y[3] = ((int64_t)a3 * G + (1 << 15)) >> 16;
        }
    }
}

// G = (wmul * m * 128 + r / 2) / r (0 for an all-zero input, whose output is 0).
static inline int64_t gain(int32_t wmul, const qvec_t *q) {
    return q->m ? ((int64_t)wmul * q->m * 128 + q->r / 2) / q->r : 0;
}

static void HOT(matvec_y)(const lm_lin_t *L, int row0, int nrows, const qvec_t *q, int32_t *acc, int64_t *y) {
    const int cb = L->cols / 4;
    const int nchunks = (L->cols + LM_TCHUNK - 1) / LM_TCHUNK;
    for (int ch = 0; ch < nchunks; ch++) {
        build_tab(q, ch);
        const int g0 = ch * LM_TG, g1 = g0 + LM_TG < cb ? g0 + LM_TG : cb, ng = g1 - g0;
        if (L->il) {
            const rows_job_t job = { (const uint32_t *)L->w + (uint32_t)(row0 / 4) * cb + g0, cb, ng, ch == 0, acc,
                                     ch == nchunks - 1 ? y : 0, y ? gain(L->wmul, q) : 0 };
            // Weights in flash (XIP, 0x10000000..): one core only. Two cores missing the XIP cache at two addresses
            // serialize on the QSPI bus and each miss gets dearer: measured slower than one core (notes/lm.md).
            const int in_xip = ((uintptr_t)L->w >> 28) == 1;
            par(rows_il, &job, nrows / 4, in_xip ? 0 : nrows / 8);
        } else {
            for (int r = 0; r < nrows; r++) {
                const uint8_t *w = L->w + (uint32_t)(row0 + r) * cb + g0;
                int32_t a = 0;
                for (int g = 0; g < ng; g++) a += tab[g][w[g]];
                acc[r] = ch == 0 ? a : acc[r] + a;
            }
        }
    }
}

static inline void matvec(const lm_lin_t *L, int row0, int nrows, const qvec_t *q, int32_t *acc) {
    matvec_y(L, row0, nrows, q, acc, 0);
}

static void lin(const lm_lin_t *L, const qvec_t *q, int64_t *y) {   // interleaved layers (every caller)
    static int32_t acc[LM_MAXN];
    matvec_y(L, 0, L->rows, q, acc, y);
}

// ----------------------------------------------------------------------------------------------- attention
// codes and max of one head slice (no pre-shift: the slice is a layer output in residual units). The division fits
// 32 bits whenever m < 2^24 (255 m < 2^32), the usual case; otherwise the 64-bit form.
static int64_t head_codes(const int64_t *x, int8_t *c) {
    int64_t m = 0;
    for (int i = 0; i < LM_D_HEAD; i++) { int64_t a = x[i] < 0 ? -x[i] : x[i]; if (a > m) m = a; }
    if (m == 0) { memset(c, 0, LM_D_HEAD); return 0; }
    if (m < (1 << 24)) {
        const uint32_t m1 = (uint32_t)m, m2 = 2 * m1;
        for (int i = 0; i < LM_D_HEAD; i++) c[i] = (int8_t)code32((int32_t)x[i], m1, m2);
    } else {
        for (int i = 0; i < LM_D_HEAD; i++) c[i] = (int8_t)code_of(x[i], m);
    }
    return m;
}

static const int alibi_shift[LM_HEADS] = LM_ALIBI_SHIFTS;

typedef struct { int layer, T, p0; const int64_t *q; int64_t *o; } heads_job_t;

// Attention core for heads lo..hi-1 (one core's share). Scratch per core: core 0 always gets lo = 0.
static void HOT(heads)(const void *ctx, int lo, int hi) {
    static int64_t s_[2][LM_WINDOW], wv_[2][LM_WINDOW];
    static int32_t e_[2][LM_WINDOW], o32_[2][LM_D_HEAD];
    const heads_job_t *J = ctx;
    const int sc = lo == 0 ? 0 : 1, layer = J->layer, T = J->T, p0 = J->p0;
    int64_t *s = s_[sc], *wv = wv_[sc];
    int32_t *e = e_[sc], *o32 = o32_[sc];
    for (int h = lo; h < hi; h++) {
        const int g = h / (LM_HEADS / LM_KV_HEADS);
        int8_t qc[LM_D_HEAD] __attribute__((aligned(4)));
        const int64_t mq = head_codes(J->q + h * LM_D_HEAD, qc);
        const int64_t Cq = (mq * LM_KQ + (1 << 15)) >> 16;
        uint32_t qe[LM_D_HEAD / 4], qo[LM_D_HEAD / 4];  // int16 pairs (q0,q2), (q1,q3) for SMLAD
        for (int d = 0; d < LM_D_HEAD; d += 4) {
            qe[d / 4] = (uint16_t)qc[d] | ((uint32_t)(uint16_t)qc[d + 2] << 16);
            qo[d / 4] = (uint16_t)qc[d + 1] | ((uint32_t)(uint16_t)qc[d + 3] << 16);
        }
        int64_t smax = INT64_MIN;
        for (int j = 0; j < T; j++) {
            const int slot_j = (p0 + j) % LM_WINDOW;
            const uint32_t *kw = (const uint32_t *)&kc[layer][slot_j][g * LM_D_HEAD];
            int32_t dot = 0;
            for (int d = 0; d < LM_D_HEAD / 4; d++) {   // 4 codes per word: SXTB16 gives (k0,k2) and (k1,k3)
                const uint32_t k4 = kw[d];
                dot = SMLAD(qe[d], SXTB16(k4), dot);
                dot = SMLAD(qo[d], SXTB16(ROR8(k4)), dot);
            }
            const int64_t a = ((int64_t)dot * km[layer][slot_j][g]) >> 12;
            const int64_t dist = T - 1 - j;
            s[j] = ((a * Cq + (1 << 23)) >> 24) - (dist << alibi_shift[h]);
            if (s[j] > smax) smax = s[j];
        }
        uint32_t S = 0;                                  // <= 256 * 65447 < 2^24
        for (int j = 0; j < T; j++) {
            const int64_t u = ((s[j] - smax) * 5909) >> 12;
            const int64_t ip = -(u >> 12);
            e[j] = ip < 31 ? EXP2[(u & 4095) >> 4] >> ip : 0;
            S += (uint32_t)e[j];
        }
        // w_j = (p_j * mv_j) >> 12 >= 0. If 127 * sum w_j < 2^31, every partial sum of w_j * vc fits int32.
        uint64_t wsum = 0;
        for (int j = 0; j < T; j++) {
            const uint32_t p = (255 * (uint32_t)e[j] + S / 2) / S;   // < 2^25: hardware UDIV
            const int64_t w = p ? ((int64_t)p * vm[layer][(p0 + j) % LM_WINDOW][g]) >> 12 : 0;
            wv[j] = w;
            wsum += (uint64_t)w;
        }
        int64_t *oh = J->o + h * LM_D_HEAD;
        if (wsum * 127 < (1ull << 31)) {
            memset(o32, 0, LM_D_HEAD * sizeof(int32_t));
            for (int j = 0; j < T; j++) {
                const int32_t w = (int32_t)wv[j];
                if (!w) continue;
                const int8_t *vj = &vc[layer][(p0 + j) % LM_WINDOW][g * LM_D_HEAD];
                for (int d = 0; d < LM_D_HEAD; d++) o32[d] += w * vj[d];
            }
            for (int d = 0; d < LM_D_HEAD; d++) oh[d] = o32[d];
        } else {
            memset(oh, 0, LM_D_HEAD * sizeof(int64_t));
            for (int j = 0; j < T; j++) {
                const int64_t w = wv[j];
                if (!w) continue;
                const int8_t *vj = &vc[layer][(p0 + j) % LM_WINDOW][g * LM_D_HEAD];
                for (int d = 0; d < LM_D_HEAD; d++) oh[d] += w * vj[d];
            }
        }
    }
}

static void HOT(attention)(int layer, const qvec_t *qx, int64_t *out) {
    static int64_t q[LM_HEADS * LM_D_HEAD], k[LM_KVD], v[LM_KVD], o[LM_HEADS * LM_D_HEAD];
    const lm_block_t *B = &lm_model.blocks[layer];
    PROF_T0();
    lin(&B->wq, qx, q);
    lin(&B->wk, qx, k);
    lin(&B->wv, qx, v);
    const int slot = lm_pos % LM_WINDOW;
    for (int g = 0; g < LM_KV_HEADS; g++) {
        km[layer][slot][g] = (int32_t)head_codes(k + g * LM_D_HEAD, &kc[layer][slot][g * LM_D_HEAD]);
        vm[layer][slot][g] = (int32_t)head_codes(v + g * LM_D_HEAD, &vc[layer][slot][g * LM_D_HEAD]);
    }
    PROF(LM_ST_QKV);
    const int T = lm_pos + 1 < LM_WINDOW ? lm_pos + 1 : LM_WINDOW;
    const heads_job_t job = { layer, T, lm_pos + 1 - T, q, o };
    par(heads, &job, LM_HEADS, LM_HEADS / 2);
    PROF(LM_ST_ATTN);
    static qvec_t qo;
    quant(o, LM_HEADS * LM_D_HEAD, &qo);
    lin(&B->wo, &qo, out);
    PROF(LM_ST_WO);
}

// ---------------------------------------------------------------------------------------------------- FFN
static void HOT(ffn)(int layer, const int64_t *x, int64_t *y) {
    const lm_block_t *B = &lm_model.blocks[layer];
    static qvec_t qx, qh;
    static int64_t h[LM_MAXN];
    static int32_t acc[LM_MAXN];
    PROF_T0();
    quant(x, LM_D, &qx);
#if LM_TREE
    int n = 0;
    for (int lvl = 0; lvl < LM_DEPTH; lvl++) {        // only the nodes on the path
        int32_t a;
        matvec(&B->nodes, (1 << lvl) - 1 + n, 1, &qx, &a);
        n = 2 * n + (a >= 0);
    }
    matvec_y(&B->leaf_in, n * LM_LEAF, LM_LEAF, &qx, acc, h);
    PROF(LM_ST_FFN1);
    for (int i = 0; i < LM_LEAF; i++) if (h[i] < 0) h[i] = 0;
    quant(h, LM_LEAF, &qh);
    lin(&B->leaf_out[n], &qh, y);
#else
    lin(&B->w1, &qx, h);
    PROF(LM_ST_FFN1);
    for (int i = 0; i < LM_HIDDEN; i++) if (h[i] < 0) h[i] = 0;
    quant(h, LM_HIDDEN, &qh);
    lin(&B->w2, &qh, y);
#endif
    PROF(LM_ST_FFN2);
}

// --------------------------------------------------------------------------------------------------- step
int HOT(lm_step)(int token) {
    static int64_t x[LM_D], y[LM_D];
    static qvec_t qx;
    token &= LM_VOCAB - 1;
    const int8_t *ec = lm_model.embed_codes + token * LM_D;
    const int64_t mul = lm_model.embed_mul[token];
    for (int i = 0; i < LM_D; i++) x[i] = (ec[i] * mul + 128) >> 8;
    for (int l = 0; l < LM_LAYERS; l++) {
        quant(x, LM_D, &qx);
        attention(l, &qx, y);
        for (int i = 0; i < LM_D; i++) x[i] += y[i];
        ffn(l, x, y);
        for (int i = 0; i < LM_D; i++) x[i] += y[i];
    }
    PROF_T0();
    quant(x, LM_D, &qx);
    lin(&lm_model.unembed, &qx, lm_logits);
    int best = 0;
    for (int i = 1; i < LM_VOCAB; i++) if (lm_logits[i] > lm_logits[best]) best = i;
    PROF(LM_ST_UNEMBED);
    lm_pos++;
    return best;
}
