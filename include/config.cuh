#pragma once

#include <types.cuh>

constexpr bool PROFILE = false;
constexpr bool PROFILEREF = false;
constexpr int N = 8 * 1024 * 1024 * 16; // 由于算法中大量使用了向量化操作，所以此处必须是 8 的倍数（8 个 __half）
constexpr int WARMUP = 2;
constexpr int NREPEATS = 128;
constexpr float TOLERANCELOOSE = 1e-6 * N;
constexpr float TOLERANCETIGHT = 1e-3;
constexpr int BINSIZE = 1024;
constexpr float COMPARE = 0.5f;
constexpr float LOWERLEVEL = 0.0;
constexpr float UPPERLEVEL = 1.0;
constexpr QuantizeType QUANTIZETYPE = SYMMETRIC_PER_CHANNEL;