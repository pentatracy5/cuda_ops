#pragma once

constexpr bool PROFILE = false;
constexpr bool PROFILEREF = false;
constexpr int N = 4 * 1024 * 1024 * 32; // 由于算法中大量使用了向量化操作，所以此处必须是4的倍数
constexpr int WARMUP = 2;
constexpr int NREPEATS = 128;
constexpr float TOLERANCE = 1e-6 * N;
constexpr int BINSIZE = 1024;
constexpr float COMPARE = 0.5f;