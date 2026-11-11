// CMSIS-NN int8 baseline (Phase 0 step 5): the same network and the same integer weights as ref_c,
// run with ARM's optimized int8 kernels and static per-layer requantization (see notes/cmsis-nn-baseline.md).
//
// Implements the ref_infer() API from ../ref_c/ref_infer.h, so the same pico_main.c and harness drive it.
// Logits are the int8 outputs of the last layer, widened to int32.
//
// Model header chosen at build time: -DS8_MODEL_HEADER="model_s8_tern80.h"
#include <stdint.h>
#include "pico.h"
#include "arm_nnfunctions.h"
#include "ref_infer.h"

#ifndef S8_MODEL_HEADER
#error "define S8_MODEL_HEADER, e.g. -DS8_MODEL_HEADER=\"model_s8_tern80.h\""
#endif
#include S8_MODEL_HEADER

#ifndef CMSIS_NN_USE_SINGLE_ROUNDING
#error "build with CMSIS_NN_USE_SINGLE_ROUNDING: the golden model (harness/golden.py) assumes it"
#endif

const char ref_model_name[] = S8_MODEL_HEADER;

#define C S8_CH

// NHWC activation buffers, ping-pong. Largest tensor is 14 x 14 x C.
static int8_t buf_a[14 * 14 * C];
static int8_t buf_b[12 * 12 * C];
static int8_t fc_a[S8_FC1_OUT], fc_b[S8_FC2_OUT];

#define SCRATCH_BYTES 2048
static int8_t scratch[SCRATCH_BYTES] __attribute__((aligned(4)));

// Per-channel quantization arrays (CMSIS-NN convs take per-channel params; all channels share one scale here).
static int32_t conv1_mult[C], conv1_shift[C], dw1_mult[C], dw1_shift[C], dw2_mult[C], dw2_shift[C];

static const cmsis_nn_dims in_dims    = {1, 16, 16, 1};
static const cmsis_nn_dims conv1_fdim = {C, 3, 3, 1};
static const cmsis_nn_dims conv1_odim = {1, 14, 14, C};
static const cmsis_nn_dims dw1_idim   = {1, 14, 14, C};
static const cmsis_nn_dims dw_fdim    = {1, 3, 3, C};
static const cmsis_nn_dims dw1_odim   = {1, 12, 12, C};
static const cmsis_nn_dims pool1_idim = {1, 12, 12, C};
static const cmsis_nn_dims pool_fdim  = {1, 2, 2, 1};
static const cmsis_nn_dims pool1_odim = {1, 6, 6, C};
static const cmsis_nn_dims dw2_idim   = {1, 6, 6, C};
static const cmsis_nn_dims dw2_odim   = {1, 4, 4, C};
static const cmsis_nn_dims pool2_idim = {1, 4, 4, C};
static const cmsis_nn_dims pool2_odim = {1, 2, 2, C};
static const cmsis_nn_dims no_bias    = {0, 0, 0, 0};

const char *const ref_stage_names[REF_N_STAGES] = {"conv1", "dw1", "pool1", "dw2", "pool2", "fc1", "fc2", "fc3"};
uint32_t ref_stage_cycles[REF_N_STAGES];

static int initialized;
int cmsis_check_scratch(void);

static void init(void) {
    int err = cmsis_check_scratch();
    if (err) panic("cmsis_nn: scratch buffer too small, need %d bytes", -err);
    for (int i = 0; i < C; i++) {
        conv1_mult[i] = S8_CONV1_MULT; conv1_shift[i] = S8_CONV1_SHIFT;
        dw1_mult[i] = S8_DW1_MULT;     dw1_shift[i] = S8_DW1_SHIFT;
        dw2_mult[i] = S8_DW2_MULT;     dw2_shift[i] = S8_DW2_SHIFT;
    }
    initialized = 1;
}

// Returns 0 on success, or a negative error. Also used once at start-up to check the scratch size.
int cmsis_check_scratch(void) {
    const cmsis_nn_conv_params cp = {0, 0, {1, 1}, {0, 0}, {1, 1}, {0, 127}};
    const cmsis_nn_dw_conv_params dp = {0, 0, 1, {1, 1}, {0, 0}, {1, 1}, {0, 127}};
    int32_t need = arm_convolve_wrapper_s8_get_buffer_size(&cp, &in_dims, &conv1_fdim, &conv1_odim);
    int32_t n2 = arm_depthwise_conv_wrapper_s8_get_buffer_size(&dp, &dw1_idim, &dw_fdim, &dw1_odim);
    int32_t n3 = arm_depthwise_conv_wrapper_s8_get_buffer_size(&dp, &dw2_idim, &dw_fdim, &dw2_odim);
    if (n2 > need) need = n2;
    if (n3 > need) need = n3;
    return need <= SCRATCH_BYTES ? 0 : -(int)need;
}

static void fc(const int8_t *in, int n_in, const int8_t *w, int n_out, int32_t mult, int32_t shift,
               int32_t act_min, int8_t *out) {
    const cmsis_nn_context ctx = {scratch, SCRATCH_BYTES};
    const cmsis_nn_fc_params p = {0, 0, 0, {act_min, 127}};
    const cmsis_nn_per_tensor_quant_params q = {mult, shift};
    const cmsis_nn_dims idim = {1, 1, 1, n_in};
    const cmsis_nn_dims fdim = {n_in, 1, 1, n_out};
    const cmsis_nn_dims odim = {1, 1, 1, n_out};
    arm_fully_connected_s8(&ctx, &p, &q, &idim, in, &fdim, w, &no_bias, NULL, &odim, out);
}

uint32_t ref_infer(const int8_t img[REF_IMG_PIXELS], int32_t logits[REF_N_CLASSES]) {
    if (!initialized) init();
    const cmsis_nn_context ctx = {scratch, SCRATCH_BYTES};
    const cmsis_nn_conv_params cp = {0, 0, {1, 1}, {0, 0}, {1, 1}, {0, 127}};
    const cmsis_nn_dw_conv_params dp = {0, 0, 1, {1, 1}, {0, 0}, {1, 1}, {0, 127}};
    const cmsis_nn_pool_params pp = {{2, 2}, {0, 0}, {0, 127}};
    cmsis_nn_per_channel_quant_params q;
    PROF_BEGIN();

    q.multiplier = conv1_mult; q.shift = conv1_shift;
    arm_convolve_wrapper_s8(&ctx, &cp, &q, &in_dims, img, &conv1_fdim, s8_conv1_w, &no_bias, NULL, &conv1_odim, buf_a);
    PROF_STAGE(ST_CONV1);

    q.multiplier = dw1_mult; q.shift = dw1_shift;
    arm_depthwise_conv_wrapper_s8(&ctx, &dp, &q, &dw1_idim, buf_a, &dw_fdim, s8_dw1_w, &no_bias, NULL, &dw1_odim, buf_b);
    PROF_STAGE(ST_DW1);
    arm_max_pool_s8(&ctx, &pp, &pool1_idim, buf_b, &pool_fdim, &pool1_odim, buf_a);
    PROF_STAGE(ST_POOL1);

    q.multiplier = dw2_mult; q.shift = dw2_shift;
    arm_depthwise_conv_wrapper_s8(&ctx, &dp, &q, &dw2_idim, buf_a, &dw_fdim, s8_dw2_w, &no_bias, NULL, &dw2_odim, buf_b);
    PROF_STAGE(ST_DW2);
    arm_max_pool_s8(&ctx, &pp, &pool2_idim, buf_b, &pool_fdim, &pool2_odim, buf_a);
    PROF_STAGE(ST_POOL2);

    fc(buf_a, 4 * C, s8_fc1_w, S8_FC1_OUT, S8_FC1_MULT, S8_FC1_SHIFT, 0, fc_a);
    PROF_STAGE(ST_FC1);
    fc(fc_a, S8_FC1_OUT, s8_fc2_w, S8_FC2_OUT, S8_FC2_MULT, S8_FC2_SHIFT, 0, fc_b);
    PROF_STAGE(ST_FC2);
    int8_t out8[REF_N_CLASSES];
    fc(fc_b, S8_FC2_OUT, s8_fc3_w, REF_N_CLASSES, S8_FC3_MULT, S8_FC3_SHIFT, -128, out8);
    PROF_STAGE(ST_FC3);

    uint32_t best = 0;
    for (uint32_t i = 0; i < REF_N_CLASSES; i++) {
        logits[i] = out8[i];
        if (out8[i] > out8[best]) best = i;
    }
    return best;
}
