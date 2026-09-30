#!/usr/bin/env bash
# Sanity check of the machine before touching any model.
# Usage: bash scripts/check_env.sh
set -u
hr(){ printf '\n== %s ==\n' "$1"; }

hr "GPU"
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version,compute_cap --format=csv
else
  echo "nvidia-smi not found - install the NVIDIA driver first (Blackwell laptop GPUs need >= 570.x)"
fi

hr "CUDA toolkit (optional for Docker path)"
command -v nvcc >/dev/null && nvcc --version | tail -1 || echo "nvcc not found (fine if you use Docker images)"

hr "Docker + NVIDIA container toolkit"
if command -v docker >/dev/null; then
  docker --version
  docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi -L 2>/dev/null \
    || echo "docker --gpus all failed: install nvidia-container-toolkit and 'sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker'"
else
  echo "docker not found"
fi

hr "Python"
python3 --version
python3 - <<'PY'
import importlib
for m in ("numpy", "mujoco", "httpx", "PIL", "yaml"):
    try:
        mod = importlib.import_module(m)
        print(f"  {m:8s} {getattr(mod, '__version__', 'ok')}")
    except Exception as e:  # noqa: BLE001
        print(f"  {m:8s} MISSING ({e.__class__.__name__}) -> pip install -e '.[dev]'")
PY

hr "Headless rendering (MuJoCo)"
for gl in egl osmesa; do
  MUJOCO_GL=$gl python3 - <<'PY' 2>/dev/null && echo "  MUJOCO_GL=$gl OK" || echo "  MUJOCO_GL=$gl failed"
import mujoco
m = mujoco.MjModel.from_xml_string("<mujoco><worldbody><light pos='0 0 1'/><geom size='0.1'/><camera name='c' pos='0 -1 0.5' xyaxes='1 0 0 0 0.5 1'/></worldbody></mujoco>")
r = mujoco.Renderer(m, 32, 32); d = mujoco.MjData(m); mujoco.mj_forward(m, d); r.update_scene(d, 'c'); r.render()
PY
done
echo "  (EGL uses the GPU; if only osmesa works: sudo apt install libegl1 libgl1 libosmesa6)"

hr "Disk / RAM"
df -h ~ | tail -1
free -g | head -2
echo "  DiffusionGemma NVFP4 weights ~18 GB in ~/.cache/huggingface"

hr "API keys"
[ -n "${TYPESAFE_API_KEY:-}" ] && echo "  TYPESAFE_API_KEY set" || echo "  TYPESAFE_API_KEY not set (export it or put it in .env)"
[ -n "${HF_TOKEN:-}" ] && echo "  HF_TOKEN set" || echo "  HF_TOKEN not set (needed if the DiffusionGemma repo is gated)"
