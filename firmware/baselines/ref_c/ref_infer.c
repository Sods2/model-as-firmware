// Plain-C reference inference for the BitNetMCU CNNMNIST layout:
//   conv3x3 1->C (8-bit) -> depthwise conv3x3 (8-bit) -> maxpool2 -> depthwise conv3x3 (8-bit) -> maxpool2
//   -> fc1 ternary (10 trits per uint16) -> fc2 4bitsym -> fc3 4bitsym -> logits
//
// Integer semantics follow BitNetMCU's MCU engine, so the results are bit-exact with it. The code
// itself is written independently for this project. Weights stay in the exported packed format
// (read from flash on the Pico); activations are int32 inside the conv stack and int8 between fc layers.
//
// The model header is chosen at build time: -DMODEL_HEADER="tern80.h"
#include <stdint.h>
#include "ref_infer.h"

#ifndef MODEL_HEADER
#error "define MODEL_HEADER, e.g. -DMODEL_HEADER=\"tern80.h\""
#endif
#include MODEL_HEADER

const char ref_model_name[] = MODEL_HEADER;

// ---- layout checks: this reference only supports the CNNMNIST layout ----
#ifndef MODEL_CNNMNIST
#error "ref_infer.c supports only the CNNMNIST model layout"
#endif
#if L2_incoming_x != 16 || L2_in_channels != 1 || L2_kernel_size != 3
#error "unexpected first conv layer"
#endif
#if L4_groups != L4_in_channels || L7_groups != L7_in_channels
#error "L4 and L7 must be depthwise"
#endif
#if L11_bitperweight != 64
#error "fc1 (L11) must be ternary (10 trits per uint16)"
#endif
#if L13_bitperweight != 4 || L15_bitperweight != 4
#error "fc2/fc3 (L13/L15) must be 4bitsym"
#endif
#if L15_outgoing_weights != REF_N_CLASSES
#error "classifier must have 10 outputs"
#endif

#define N_CH        L7_out_channels
#define FEAT_PER_CH (L9_outgoing_x * L9_outgoing_y)
#define N_FEAT      (N_CH * FEAT_PER_CH)
#if N_FEAT != L11_incoming_weights
#error "flattened conv output does not match fc1 input"
#endif

#define CONV_SHIFT 4  // fixed right shift after each conv, as in BitNetMCU's MCU engine

// 3x3 valid convolution of one n x n channel, then ReLU and >> CONV_SHIFT. Output is (n-2) x (n-2).
static void conv3x3_relu(const int32_t *in, int n, const int8_t *k, int32_t *out) {
    const int32_t k0 = k[0], k1 = k[1], k2 = k[2];
    const int32_t k3 = k[3], k4 = k[4], k5 = k[5];
    const int32_t k6 = k[6], k7 = k[7], k8 = k[8];
    const int m = n - 2;
    for (int y = 0; y < m; y++) {
        const int32_t *r0 = in + y * n;
        const int32_t *r1 = r0 + n;
        const int32_t *r2 = r1 + n;
        for (int x = 0; x < m; x++) {
            int32_t s = k0 * r0[x] + k1 * r0[x + 1] + k2 * r0[x + 2]
                      + k3 * r1[x] + k4 * r1[x + 1] + k5 * r1[x + 2]
                      + k6 * r2[x] + k7 * r2[x + 1] + k8 * r2[x + 2];
            *out++ = s < 0 ? 0 : (s >> CONV_SHIFT);
        }
    }
}

// 2x2 max pool, stride 2, of one n x n channel. Output is (n/2) x (n/2).
static void maxpool2(const int32_t *in, int n, int32_t *out) {
    const int m = n / 2;
    for (int y = 0; y < m; y++) {
        const int32_t *r0 = in + 2 * y * n;
        const int32_t *r1 = r0 + n;
        for (int x = 0; x < m; x++) {
            int32_t a = r0[2 * x], b = r0[2 * x + 1], c = r1[2 * x], d = r1[2 * x + 1];
            int32_t ab = a > b ? a : b, cd = c > d ? c : d;
            *out++ = ab > cd ? ab : cd;
        }
    }
}

// ReLU + dynamic power-of-two requantization to int8 (0..127).
// The shift is the smallest one that brings the maximum below 128, with round-half-up.
static void relu_norm(const int32_t *in, int8_t *out, int n) {
    int32_t mx = -INT32_MAX;
    for (int i = 0; i < n; i++) if (in[i] > mx) mx = in[i];
    int shift = 0;
    for (int32_t s = mx >> 7; s > 0; s >>= 1) shift++;
    const int32_t rnd = (1 << shift) >> 1;
    for (int i = 0; i < n; i++) {
        int32_t v = in[i];
        if (v < 0) { out[i] = 0; continue; }
        v = (v + rnd) >> shift;
        out[i] = (int8_t)(v > 127 ? 127 : v);
    }
}

// Ternary fully connected layer. Each uint16 holds 10 trits as a base-3 fraction:
// multiply by 3, bits 17:16 give the digit (0 -> +1, 1 -> -1, 2 -> 0), keep the low 16 bits.
static void fc_ternary(const int8_t *in, const uint16_t *w, int n_in, int n_out, int32_t *out) {
    for (int o = 0; o < n_out; o++) {
        const int8_t *a = in;
        int32_t sum = 0;
        for (int k = 0; k < n_in; k += 10) {
            uint32_t c = *w++;
            for (int j = 0; j < 10; j++) {
                c *= 3;
                uint32_t d = (c >> 16) & 3;
                if (d == 0) sum += *a;
                else if (d == 1) sum -= *a;
                a++;
                c &= 0xFFFF;
            }
        }
        out[o] = sum;
    }
}

// 4bitsym fully connected layer. 8 nibbles per uint32, most significant first.
// Nibble: bit 3 = sign (1 = negative), bits 2:0 = m, weight magnitude = 2*m + 1 (1, 3, ..., 15).
static void fc_4bitsym(const int8_t *in, const uint32_t *w, int n_in, int n_out, int32_t *out) {
    for (int o = 0; o < n_out; o++) {
        const int8_t *a = in;
        int32_t sum = 0;
        for (int k = 0; k < n_in; k += 8) {
            uint32_t word = *w++;
            for (int j = 0; j < 8; j++) {
                uint32_t nib = word >> 28;
                int32_t v = (int32_t)(2 * (nib & 7) + 1) * *a++;
                sum += (nib & 8) ? -v : v;
                word <<= 4;
            }
        }
        out[o] = sum;
    }
}

const int ref_n_stages = 8;
const char *const ref_stage_names[REF_MAX_STAGES] = {"conv1", "dw1", "pool1", "dw2", "pool2", "fc1", "fc2", "fc3"};
uint32_t ref_stage_cycles[REF_MAX_STAGES];

// Static buffers: the Pico's default stack is small.
static int32_t buf_a[16 * 16], buf_b[16 * 16];
static int32_t feat[N_FEAT];
static int32_t acc[MAX_N_ACTIVATIONS];
static int8_t  act[MAX_N_ACTIVATIONS];

uint32_t ref_infer(const int8_t img[REF_IMG_PIXELS], int32_t logits[REF_N_CLASSES]) {
    PROF_BEGIN();

    // Conv stack, one channel at a time (all convs after the first are depthwise).
    for (int c = 0; c < N_CH; c++) {
        for (int i = 0; i < 16 * 16; i++) buf_a[i] = img[i];
        conv3x3_relu(buf_a, L2_incoming_x, L2_weights + 9 * c, buf_b);   // 16 -> 14
        PROF_STAGE(ST_CONV1);
        conv3x3_relu(buf_b, L4_incoming_x, L4_weights + 9 * c, buf_a);   // 14 -> 12
        PROF_STAGE(ST_DW1);
        maxpool2(buf_a, L6_incoming_x, buf_b);                             // 12 -> 6
        PROF_STAGE(ST_POOL1);
        conv3x3_relu(buf_b, L7_incoming_x, L7_weights + 9 * c, buf_a);   //  6 -> 4
        PROF_STAGE(ST_DW2);
        maxpool2(buf_a, L9_incoming_x, feat + FEAT_PER_CH * c);            //  4 -> 2
        PROF_STAGE(ST_POOL2);
    }
    relu_norm(feat, act, N_FEAT);
    fc_ternary(act, L11_weights, L11_incoming_weights, L11_outgoing_weights, acc);
    PROF_STAGE(ST_FC1);

    relu_norm(acc, act, L11_outgoing_weights);
    fc_4bitsym(act, L13_weights, L13_incoming_weights, L13_outgoing_weights, acc);
    PROF_STAGE(ST_FC2);

    relu_norm(acc, act, L13_outgoing_weights);
    fc_4bitsym(act, L15_weights, L15_incoming_weights, L15_outgoing_weights, logits);
    PROF_STAGE(ST_FC3);

    uint32_t best = 0;
    for (uint32_t i = 1; i < REF_N_CLASSES; i++) if (logits[i] > logits[best]) best = i;
    return best;
}
