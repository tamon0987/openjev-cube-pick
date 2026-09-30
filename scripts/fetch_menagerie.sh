#!/usr/bin/env bash
# Optional: fetch the SO-ARM100 MJCF + meshes from mujoco_menagerie (Apache-2.0) if you want to
# run the same experiment on SO-101 class hardware. The default scene already uses the official
# ROBOTIS OMX model vendored under dlb/sim/assets/omx.
set -euo pipefail
DEST="${1:-third_party/menagerie}"
mkdir -p "$DEST"
cd "$DEST"
if [ ! -d mujoco_menagerie ]; then
  git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git
  cd mujoco_menagerie && git sparse-checkout set trs_so_arm100 && cd ..
fi
echo "SO-ARM100 model at $DEST/mujoco_menagerie/trs_so_arm100/scene.xml"
echo "Joints: Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw ; keyframes: home, rest"
