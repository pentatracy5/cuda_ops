#pragma once

constexpr bool PROFILE = false;
constexpr bool PROFILEREF = false;
constexpr int N = 4 * 1024 * 1024 * 32;
constexpr int WARMUP = 2;
constexpr int NREPEATS = 128;
constexpr float TOLERANCE = 1e-6 * N;