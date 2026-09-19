#!/usr/bin/env bash
set +x

input_image=${1:-'input.jpeg'}

echo "Encoding ${input_image}"

cargo build \
  --release \
  --features face-detect

./target/release/encmars \
   ${input_image} \
   output.mars \
  --human-adaptive \
  --lambda 100 \
  --debug-regions regions.png \
  --color eyes \
  --outside-min-size 16 \
  --scrfd-model ./models/det_10g.onnx

./target/release/decmars \
   output.mars \
   output.jpeg
