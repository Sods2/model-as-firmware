// Plain-C reference inference for the BitNetMCU CNNMNIST layout.
// This is the correctness anchor: every faster build must give the same logits.
#ifndef REF_INFER_H
#define REF_INFER_H

#include <stdint.h>

#define REF_IMG_PIXELS 256  // 16x16 int8 input
#define REF_N_CLASSES  10

// Runs one image. Writes the 10 raw int32 logits and returns the predicted
// class (index of the first maximum logit).
uint32_t ref_infer(const int8_t img[REF_IMG_PIXELS], int32_t logits[REF_N_CLASSES]);

// Name of the model header compiled in, for logging.
extern const char ref_model_name[];

// Per-stage cycle counts of the last ref_infer() call (all zero on the PC build).
// Each build defines its own stages (ref_n_stages of them, at most REF_MAX_STAGES). For the CNNMNIST
// builds they are the same 8 (conv1 dw1 pool1 dw2 pool2 fc1 fc2 fc3) so builds can be compared;
// requantization/normalization is counted in the stage that consumes it.
#define REF_MAX_STAGES 16
extern const int ref_n_stages;
extern const char *const ref_stage_names[REF_MAX_STAGES];
extern uint32_t ref_stage_cycles[REF_MAX_STAGES];
enum { ST_CONV1, ST_DW1, ST_POOL1, ST_DW2, ST_POOL2, ST_FC1, ST_FC2, ST_FC3 };

#if defined(PICO_ON_DEVICE) && PICO_ON_DEVICE
static inline uint32_t ref_cycles(void) { return *(volatile uint32_t *)0xE0001004u; }  // DWT_CYCCNT
#else
static inline uint32_t ref_cycles(void) { return 0; }
#endif

// Stage profiling costs a few % (one counter read per stage, per channel in ref_c), so it is
// compiled in only with -DREF_PROFILE=1 (CMake option PROFILE_STAGES). Timed runs use builds without it.
#if defined(REF_PROFILE) && REF_PROFILE
#define PROF_BEGIN()     uint32_t prof_t_ = ref_cycles();     for (int prof_i_ = 0; prof_i_ < REF_MAX_STAGES; prof_i_++) ref_stage_cycles[prof_i_] = 0
#define PROF_STAGE(id)     do { uint32_t prof_t1_ = ref_cycles(); ref_stage_cycles[id] += prof_t1_ - prof_t_; prof_t_ = prof_t1_; } while (0)
#else
#define PROF_BEGIN() ((void)0)
#define PROF_STAGE(id) ((void)0)
#endif

#endif
