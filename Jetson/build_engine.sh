#!/usr/bin/env bash
# Step C (run on the JETSON): compile the ONNX model into a TensorRT FP16 engine.
#
#   bash Jetson/build_engine.sh
#
# trtexec ships with JetPack. If it is not on PATH, the usual location is
# /usr/src/tensorrt/bin/trtexec — the script checks both.
#
# FP16 is safe here because the numerically sensitive frontend (STFT/log)
# stays at higher precision automatically; validate_engine.py proves the
# engine output matches the laptop reference within 1%.
set -euo pipefail

ONNX="${1:-mobilenetv2_drone.onnx}"
ENGINE="${2:-drone_fp16.engine}"

TRTEXEC="$(command -v trtexec || echo /usr/src/tensorrt/bin/trtexec)"
if [ ! -x "$TRTEXEC" ]; then
  echo "trtexec not found — is JetPack / TensorRT installed?" >&2
  exit 1
fi

# Fixed batch=1 (live detection runs one 1-second window at a time).
"$TRTEXEC" \
  --onnx="$ONNX" \
  --fp16 \
  --minShapes=waveform:1x16000 \
  --optShapes=waveform:1x16000 \
  --maxShapes=waveform:1x16000 \
  --saveEngine="$ENGINE"

echo "engine written to $ENGINE — next: python3 validate_engine.py --engine $ENGINE"

# FP32 fallback (only needed if validate_engine.py ever fails the 1% check):
#   "$TRTEXEC" --onnx="$ONNX" --minShapes=waveform:1x16000 \
#     --optShapes=waveform:1x16000 --maxShapes=waveform:1x16000 \
#     --saveEngine=drone_fp32.engine
