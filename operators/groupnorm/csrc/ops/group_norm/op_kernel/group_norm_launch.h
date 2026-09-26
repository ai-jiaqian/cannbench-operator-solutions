/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

/*!
 * \file group_norm_launch.h
 * \brief GroupNorm launch declarations (g++ visible) + tiling struct
 */

#ifndef GROUP_NORM_LAUNCH_H
#define GROUP_NORM_LAUNCH_H

#include <cstdint>

#ifndef GM_ADDR
#define GM_ADDR void*
#endif

// Tiling parameters shared between host tiling calc (bisheng TU) and plugin (g++ TU)
struct GroupNormTilingParams {
    int64_t tasks1;    // N*G*P1
    int64_t tasks2;    // N*G*P2
    int64_t shardLen1; // ceil(slabLen/P1)
    int64_t slabLen;   // (C/G)*S
    int64_t S;         // spatial product
    int64_t blocks1;
    int64_t blocks2;
    int32_t P1;
    int32_t P2;
    int32_t G;    // num_groups
    int32_t Cp;   // C/G
    int32_t cs;   // ceil(Cp/P2)
    int32_t flat; // 1: S==1 && slabLen<=2048 whole-slab vector path (kernel B only)
    int32_t modeS;    // 1: single-kernel whole-group path (no workspace)
    uint32_t TILES;   // tile size (elements) for the single-kernel path
    int64_t blocksS;
    int64_t tasksS;   // modeS task count under batch_groups: ceil(NG/gpt)
    int32_t gpt;      // modeS groups per task (1 = incumbent one-group tasks)
    uint32_t TILE1;
    uint32_t TILE2;
    uint32_t W;      // weight buffer capacity in floats (g32/b32/aV/bV/wT each)
    float invCnt;    // 1/slabLen (host double -> float)
    int32_t modeF;     // 1: fused single-launch mode (z1 solve 8e372571...)
    int32_t P1F;       // mode-F shard count per group
    int64_t shardLenF; // ceil(slabLen/P1F)
    int64_t tasksF;    // NG*P1F
    int64_t blocksF;
    uint32_t TILEF;
    uint32_t WF;       // mode-F weight buffer capacity in floats
    int64_t cntFloats; // mode-F counter workspace floats (4*NG)
    int32_t expS;      // 1: kernel-S non-flat concatenated-segment affine expansion
                       // (z002 sc1_kernel_s_expanded_affine non-flat form; z2 solve
                       // 154df1a37717...): S>1 && S%8==0 && slabLen<=4096 && UB fit.
    int32_t leanS;     // 1: kernel-S lean merged-scratch inventory (z002
                       // sc2_lean_modes_memory_plan; z2 solve cde40635f38bf4...
                       // via models/gn_lower_z2_iter2.py): modeS && S>1 && fp32
                       // && slabLen>=49152 (disjoint from the expS gate).
    int32_t resS;      // 1: kernel-S slab-resident single-pass (z003 sc1
                       // kernel_s_resident_memory_plan; z3 solve b8810356f733...
                       // via models/gn_lower_z3_iter1.py): modeS && 16-bit &&
                       // (flat || (S*ts)%32==0) && slabLen*ts<=65536 && UB fit.
    int32_t winS;      // z3 sc2 weight_window_gate (z3 Lower iteration-2 solve
                       // c3cf86e8585b... via models/gn_lower_z3_iter2.py): 1 =
                       // per-task full-coverage gamma/beta window in kernel S
                       // (modeS && !flat && gpt>1 && gpt>=G && Cp%8==0 &&
                       // G*Cp<=W); reuses wQg/wQb/g32/b32 (no new UB bytes).
    int32_t winCh;     // z3 sc2 window channel count (G*Cp when winS == 1).
};

// Tiling calculation (defined in bisheng-compiled kernel TU)
GroupNormTilingParams calc_group_norm_tiling(int64_t N, int64_t C, int64_t G, int64_t S, int64_t typeSize);

extern "C" {

void launch_group_norm_stats_float(GM_ADDR x, GM_ADDR ws,
    int64_t tasks1, int32_t P1, int64_t shardLen1, int64_t slabLen,
    uint32_t tile1, int64_t numBlocks, void* stream);

void launch_group_norm_stats_half(GM_ADDR x, GM_ADDR ws,
    int64_t tasks1, int32_t P1, int64_t shardLen1, int64_t slabLen,
    uint32_t tile1, int64_t numBlocks, void* stream);

void launch_group_norm_stats_bf16(GM_ADDR x, GM_ADDR ws,
    int64_t tasks1, int32_t P1, int64_t shardLen1, int64_t slabLen,
    uint32_t tile1, int64_t numBlocks, void* stream);

void launch_group_norm_norm_float(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR ws,
    int64_t tasks1, int64_t tasks2, int32_t P1, int32_t P2, int32_t G, int32_t Cp, int32_t cs,
    int64_t slabLen, int64_t S, uint32_t Wcap, uint32_t tile2, int32_t flat,
    float invCnt, float eps, int64_t numBlocks, void* stream);

void launch_group_norm_norm_half(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR ws,
    int64_t tasks1, int64_t tasks2, int32_t P1, int32_t P2, int32_t G, int32_t Cp, int32_t cs,
    int64_t slabLen, int64_t S, uint32_t Wcap, uint32_t tile2, int32_t flat,
    float invCnt, float eps, int64_t numBlocks, void* stream);

void launch_group_norm_norm_bf16(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR ws,
    int64_t tasks1, int64_t tasks2, int32_t P1, int32_t P2, int32_t G, int32_t Cp, int32_t cs,
    int64_t slabLen, int64_t S, uint32_t Wcap, uint32_t tile2, int32_t flat,
    float invCnt, float eps, int64_t numBlocks, void* stream);

void launch_group_norm_single_float(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y,
    int64_t tasks, int64_t NGtotal, int32_t gpt, int32_t G, int32_t Cp, int64_t slabLen, int64_t S,
    uint32_t Wcap, uint32_t tileS, int32_t expS, int32_t leanS, int32_t resS,
    int32_t winS, int32_t winCh, float invCnt, float eps, int64_t numBlocks, void* stream);

void launch_group_norm_single_half(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y,
    int64_t tasks, int64_t NGtotal, int32_t gpt, int32_t G, int32_t Cp, int64_t slabLen, int64_t S,
    uint32_t Wcap, uint32_t tileS, int32_t expS, int32_t leanS, int32_t resS,
    int32_t winS, int32_t winCh, float invCnt, float eps, int64_t numBlocks, void* stream);

void launch_group_norm_single_bf16(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y,
    int64_t tasks, int64_t NGtotal, int32_t gpt, int32_t G, int32_t Cp, int64_t slabLen, int64_t S,
    uint32_t Wcap, uint32_t tileS, int32_t expS, int32_t leanS, int32_t resS,
    int32_t winS, int32_t winCh, float invCnt, float eps, int64_t numBlocks, void* stream);

// mode F (z1 solve 8e372571b661a71e...): fused single-launch, per-task shard
// stats + atomic publish + bounded poll + per-channel normalize.
void launch_group_norm_fused_float(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y,
    GM_ADDR cnt, int64_t tasksF, int32_t P1F, int32_t G, int32_t Cp,
    int64_t shardLenF, int64_t slabLen, int64_t S, uint32_t WcapF, uint32_t tileF,
    float invCnt, float eps, int64_t pollBudget, int64_t numBlocks, void* stream);

void launch_group_norm_fused_half(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y,
    GM_ADDR cnt, int64_t tasksF, int32_t P1F, int32_t G, int32_t Cp,
    int64_t shardLenF, int64_t slabLen, int64_t S, uint32_t WcapF, uint32_t tileF,
    float invCnt, float eps, int64_t pollBudget, int64_t numBlocks, void* stream);

void launch_group_norm_fused_bf16(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y,
    GM_ADDR cnt, int64_t tasksF, int32_t P1F, int32_t G, int32_t Cp,
    int64_t shardLenF, int64_t slabLen, int64_t S, uint32_t WcapF, uint32_t tileF,
    float invCnt, float eps, int64_t pollBudget, int64_t numBlocks, void* stream);
}

#endif // GROUP_NORM_LAUNCH_H
