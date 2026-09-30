# ノートPC（RTX PRO 5000 Blackwell 24 GB）セットアップメモ

想定 OS: Ubuntu 24.04（デュアルブート可）。Windows + WSL2 でも Docker 経由なら概ね同じですが、EGL 描画と VRAM の見え方が変わるので Ubuntu ネイティブを推奨します。

## 1. ドライバ

- Blackwell 世代のノート GPU は **NVIDIA ドライバ 570 系以降**が必要。`ubuntu-drivers devices` で提案されるものより新しい版が要る場合は `graphics-drivers` PPA。
- Secure Boot 有効なら MOK 登録が必要。
- 確認: `nvidia-smi` に `RTX PRO 5000 Blackwell` と `24 GB`、`compute_cap 12.0`。

## 2. VRAM の使い方

24 GB は openjev の公称最小値と同じで余裕がありません。

- ディスプレイ出力を iGPU（ハイブリッド / Optimus）にして、dGPU を計算専用にする。`nvidia-smi` の "Display" 行と `Xorg` / `gnome-shell` のプロセスが dGPU に載っていないことを確認。
- サーバは 1 つだけ起動する（openjev と djev を同時に立てない）。
- `--max-model-len` は 8192 程度に下げる（状態 JSON + 2 画像で数千トークン以内）。
- 電源設定を「パフォーマンス」に。AC 接続時のみ実験（バッテリー時はクロックが落ちてレイテンシが 2 倍以上ぶれる）。

## 3. Docker

```bash
sudo apt install docker.io
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
sudo usermod -aG docker $USER   # 再ログイン
```

## 4. Python / MuJoCo

```bash
sudo apt install libegl1 libgl1 libosmesa6
curl -LsSf https://astral.sh/uv/install.sh | sh   # uv が未導入なら
```

Python の仮想環境は、ROS 2 の `rclpy` を使うため [real_robot.md](real_robot.md) の 2. の手順で作る。作ったあと、描画を確認する:

```bash
MUJOCO_GL=egl uv run python -c "import mujoco; print(mujoco.__version__)"
```

`MUJOCO_GL=egl` で `EGLError` が出る場合は `libnvidia-egl-*` が入っているか、`__EGL_VENDOR_LIBRARY_FILENAMES` が NVIDIA を指しているかを確認。ダメなら `MUJOCO_GL=osmesa`（CPU 描画、遅い）で進めて後で直す。

## 5. 作業の進め方（推奨）

- `tmux` で 3 ペイン: サーバログ（`docker logs -f`）、`nvidia-smi -l 2`、コマンド。
- `.env` を `set -a; source .env; set +a` で読み込む。
