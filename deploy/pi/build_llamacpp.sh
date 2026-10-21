#!/usr/bin/env bash
# Build llama.cpp inside Raspberry Pi OS, targeting Raspberry Pi 5 Cortex-A76.
# GGML_NATIVE is OFF so Graviton SVE is not baked in (that would fail on a real Pi).
set -euo pipefail
LLAMA_SRC="${LLAMA_SRC:-/llavamod/llama.cpp}"
BUILD_DIR="${BUILD_DIR:-/llavamod/raspberry_pi_os_eval/llama.cpp-build}"
# Pi 5: Cortex-A76, ARMv8.2-A + CRC/crypto/FP16/RCPC/DotProd. No SVE, no i8mm.
ARM_ARCH="${ARM_ARCH:-armv8.2-a+crc+crypto+fp16+rcpc+dotprod}"

cmake -S "${LLAMA_SRC}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=OFF \
  -DGGML_CPU_ARM_ARCH="${ARM_ARCH}" \
  -DGGML_OPENMP=ON \
  -DGGML_BLAS=ON \
  -DGGML_BLAS_VENDOR=OpenBLAS

cmake --build "${BUILD_DIR}" -j"$(nproc)" --target llama-server llama-mtmd-cli llama-quantize

echo "Built:"
ls -lh "${BUILD_DIR}/bin/llama-server" "${BUILD_DIR}/bin/llama-mtmd-cli"
# Record compile flags for the paper / another AI
cmake -LA -N "${BUILD_DIR}" 2>/dev/null | grep -E 'GGML_NATIVE|GGML_CPU_ARM|GGML_BLAS' || true
