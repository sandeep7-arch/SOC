#!/usr/bin/env bash

# Compare NNUE Model A against either Model B OR an Elo-capped Stockfish.
# Usage:
#   ./match_models_flexible.sh [model_a.bin] [model_b.bin]
#
# Optional environment overrides:
#   COMPARE_WITH_STOCKFISH=true           # Set to true to replace Model B with Stockfish!
#   STOCKFISH_CMD=stockfish               # Path to stockfish binary
#   STOCKFISH_ELO=1500                    # Cap Stockfish at an exact Elo (1320 - 3190)
#   CUTECHESS_CMD=./cutechess-cli
#   ENGINE_SCRIPT=uci.py
#   SOC_NATIVE_ENGINE_PATH=./native_engine.so
#   OUTPUT_PGN=checkpoint_battle_results.pgn
#   BUILD_NATIVE=1                         # Rebuild native_engine.so before the match

set -euo pipefail

CUTECHESS_CMD="${CUTECHESS_CMD:-cutechess-cli}"
ENGINE_SCRIPT="${ENGINE_SCRIPT:-uci.py}"
NATIVE_ENGINE="${SOC_NATIVE_ENGINE_PATH:-./native_engine.so}"
OUTPUT_PGN="${OUTPUT_PGN:-checkpoint_battle_results.pgn}"
BUILD_NATIVE="${BUILD_NATIVE:-1}"

# Toggle mode: set to 'true' to swap Model B out for Stockfish
COMPARE_WITH_STOCKFISH="${COMPARE_WITH_STOCKFISH:-true}"
STOCKFISH_CMD="${STOCKFISH_CMD:-stockfish}"

# Set Stockfish Elo threshold (Valid range is 1320 to 3190)
STOCKFISH_ELO="${STOCKFISH_ELO:-1400}" 

MODEL_A="${1:-exports/nnue_inference_m5.bin}"
MODEL_B="${2:-exports/nnue_inference_m5.bin}"

ENGINE_A_NAME="${ENGINE_A_NAME:-NNUE_Model_A}"
ENGINE_B_NAME="${ENGINE_B_NAME:-NNUE_Model_B}"

TOTAL_GAMES="${TOTAL_GAMES:-5}"
CONCURRENCY="${CONCURRENCY:-1}"
TIME_CONTROL="${TIME_CONTROL:-30+1}"

echo "======================================================================"
echo "Initializing Cutechess NNUE model comparison"
echo "======================================================================"

echo -n "Checking cutechess-cli... "
if [[ -x "$CUTECHESS_CMD" ]]; then
  CUTECHESS_EXEC="$CUTECHESS_CMD"
elif command -v "$CUTECHESS_CMD" >/dev/null 2>&1; then
  CUTECHESS_EXEC="$CUTECHESS_CMD"
else
  echo "missing"
  echo "ERROR: '$CUTECHESS_CMD' was not found or is not executable."
  exit 1
fi
echo "found"

echo -n "Checking UCI script... "
if [[ ! -f "$ENGINE_SCRIPT" ]]; then
  echo "missing"
  echo "ERROR: '$ENGINE_SCRIPT' was not found in $(pwd)."
  exit 1
fi
echo "found"

if [[ "$BUILD_NATIVE" == "1" ]]; then
  echo "Rebuilding native engine before match..."
  ./build_native.sh
fi

echo -n "Checking native engine library... "
if [[ ! -f "$NATIVE_ENGINE" ]]; then
  echo "missing"
  echo "ERROR: native engine library '$NATIVE_ENGINE' was not found."
  exit 1
fi
echo "found"

mapfile -t NEWER_NATIVE_SOURCES < <(
  find search core nnue -type f \( -name '*.cpp' -o -name '*.hpp' \) -newer "$NATIVE_ENGINE" -print | head -n 5
)
if (( ${#NEWER_NATIVE_SOURCES[@]} > 0 )); then
  echo "ERROR: native engine library '$NATIVE_ENGINE' is older than native source files."
  echo "Rebuild first with './build_native.sh' or run this script with BUILD_NATIVE=1."
  printf '  newer: %s\n' "${NEWER_NATIVE_SOURCES[@]}"
  exit 1
fi

echo -n "Checking model A... "
if [[ ! -f "$MODEL_A" ]]; then
  echo "missing"
  echo "ERROR: model A '$MODEL_A' was not found."
  exit 1
fi
echo "found"

# Set up Engine B configuration dynamically based on the toggle flag
ENGINE_B_ARGS=()
if [[ "$COMPARE_WITH_STOCKFISH" == "true" ]]; then
  echo -n "Checking Stockfish baseline... "
  if command -v "$STOCKFISH_CMD" >/dev/null 2>&1 || [[ -x "$STOCKFISH_CMD" ]]; then
    echo "found"
  else
    echo "missing"
    echo "ERROR: Stockfish command '$STOCKFISH_CMD' was not found."
    exit 1
  fi
  
  ENGINE_B_NAME="Stockfish_Elo_${STOCKFISH_ELO}"
  ENGINE_B_ARGS=(
    -engine name="$ENGINE_B_NAME" cmd="$STOCKFISH_CMD"
    option.UCI_LimitStrength=true
    option.UCI_Elo="$STOCKFISH_ELO"
    proto=uci
  )
else
  echo -n "Checking model B... "
  if [[ ! -f "$MODEL_B" ]]; then
    echo "missing"
    echo "ERROR: model B '$MODEL_B' was not found."
    exit 1
  fi
  echo "found"
  
  ENGINE_B_ARGS=(
    -engine name="$ENGINE_B_NAME" cmd="env"
    arg="SOC_MODEL_PATH=$MODEL_B"
    arg="SOC_NATIVE_ENGINE_PATH=$NATIVE_ENGINE"
    arg="python3" arg="-u" arg="$ENGINE_SCRIPT"
    dir="." proto=uci
  )
fi

if [[ -f "$OUTPUT_PGN" ]]; then
  BACKUP="${OUTPUT_PGN}.$(date +%Y%m%d_%H%M%S).bak"
  echo "Archiving existing PGN to $BACKUP"
  mv "$OUTPUT_PGN" "$BACKUP"
fi

echo "======================================================================"
echo "Match parameters"
echo "  Engine A:     $ENGINE_A_NAME"
echo "  Model A:      $MODEL_A"
echo "  Engine B:     $ENGINE_B_NAME"
if [[ "$COMPARE_WITH_STOCKFISH" == "true" ]]; then
  echo "  Mode:         Testing against Stockfish Capped at Elo $STOCKFISH_ELO"
else
  echo "  Model B:      $MODEL_B"
  echo "  Mode:         Testing NNUE Model A vs NNUE Model B"
fi
echo "  Games:        $TOTAL_GAMES with color repeat"
echo "  Concurrency:  $CONCURRENCY"
echo "  Time control: $TIME_CONTROL"
echo "  PGN output:   $OUTPUT_PGN"
echo "======================================================================"

set +e

"$CUTECHESS_EXEC" \
  -tournament round-robin \
  -concurrency "$CONCURRENCY" \
  -games "$TOTAL_GAMES" \
  -repeat \
  -recover \
  -pgnout "$OUTPUT_PGN" \
  -engine name="$ENGINE_A_NAME" cmd="env" \
    arg="SOC_MODEL_PATH=$MODEL_A" \
    arg="SOC_NATIVE_ENGINE_PATH=$NATIVE_ENGINE" \
    arg="python3" arg="-u" arg="$ENGINE_SCRIPT" \
    dir="." proto=uci \
  "${ENGINE_B_ARGS[@]}" \
  -each tc="$TIME_CONTROL" \
  -resign movecount=3 score=600 \
  -draw movenumber=40 movecount=5 score=10

TOURNEY_STATUS=$?
set -e

echo "----------------------------------------------------------------------"
echo "Tournament finished with exit code $TOURNEY_STATUS"

if [[ -f "$OUTPUT_PGN" ]]; then
  echo "PGN written to: $OUTPUT_PGN"
  echo "Games recorded: $(grep -c '^\[Event ' "$OUTPUT_PGN" || true)"
  echo "Draws recorded: $(grep -c '^\[Result \"1/2-1/2\"\]' "$OUTPUT_PGN" || true)"
  echo
  grep -E 'Score of|Finished match|Finished game|Warning|Error' "$OUTPUT_PGN" || true
fi

exit "$TOURNEY_STATUS"
