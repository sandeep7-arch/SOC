#!/usr/bin/env bash
set -euo pipefail

CXX="${CXX:-g++}"
CXXFLAGS=(
  -std=c++17
  -O3
  -DNDEBUG
  -mavx2
  -mfma
  -Icore
  -Innue
)

echo "[build] native_engine.so"
"$CXX" "${CXXFLAGS[@]}" -fPIC -shared \
  search/native_engine_shell.cpp \
  search/native_search.cpp \
  search/evaluator_core.cpp \
  -o native_engine.so

echo "[build] nnue_trainer"
"$CXX" "${CXXFLAGS[@]}" -fopenmp \
  nnue/main.cpp \
  -o nnue_trainer

echo "[build] done"
