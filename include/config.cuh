#pragma once

constexpr int THREADS_PER_BLOCK = 1024;
constexpr int N = 4096 * 4096 * 8;
constexpr int WARMUP = 2;
constexpr int NREPEATS = 4096;
constexpr float TOLERANCE = 1e-6;