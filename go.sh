#!/usr/bin/env bash
set +x

f=${1:-'input.jpeg'}

ext="${f##*.}"
name="$(basename "${f%.*}")"

echo "Encoding '${f}'"

cp "${f}" "./images/${name}_i.${ext}"

cargo build \
  --release \
  --features face-detect \
  --quiet

./target/release/encmars \
   ${f} \
   images/${name}_m.mars \
  --human-adaptive \
  --lambda 50 \
  --color eyes \
  --min-size 8 \
  --max-size 128 \
  --outside-min-size-ramp \
  --desaturate 0.3 \
  --desaturate-ramp 0 \
  --eye-lambda-scale 0.1 \
  --scrfd-model ./models/det_10g.onnx

echo ''
rm -rf images/progress

./target/release/decmars \
   images/${name}_m.mars \
   images/${name}_o.${ext} \
   --progression images/progress
