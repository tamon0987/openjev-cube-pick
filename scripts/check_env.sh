#!/usr/bin/env bash
# Sanity check of the machine before touching any model.
# Usage: bash scripts/check_env.sh   (from the repository root; uses .venv if it exists)
set -u
PY=python3
[ -x .venv/bin/python ] && PY=.venv/bin/python
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

hr "Python ($PY)"
[ "$PY" = python3 ] && echo "  .venv not found - create it first (docs/real_robot.md, section 2)"
$PY --version
$PY - <<'PY'
import importlib
for m in ("numpy", "mujoco", "httpx", "PIL", "yaml", "cv2", "onnxruntime", "rclpy"):
    try:
        mod = importlib.import_module(m)
        print(f"  {m:11s} {getattr(mod, '__version__', 'ok')}")
    except Exception as e:  # noqa: BLE001
        hint = "source /opt/ros/jazzy/setup.bash first" if m == "rclpy" else "pip install -e '.[real,voice,dev]'"
        print(f"  {m:11s} MISSING ({e.__class__.__name__}) -> {hint}")
PY

hr "Headless rendering (MuJoCo)"
for gl in egl osmesa; do
  MUJOCO_GL=$gl $PY - <<'PY' 2>/dev/null && echo "  MUJOCO_GL=$gl OK" || echo "  MUJOCO_GL=$gl failed"
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
if grep -qE '^OPENAI_API_KEY=.+' .env 2>/dev/null || [ -n "${OPENAI_API_KEY:-}" ]; then
  echo "  OPENAI_API_KEY set"
else
  echo "  OPENAI_API_KEY not set (required: cp .env.example .env and fill it in)"
fi
