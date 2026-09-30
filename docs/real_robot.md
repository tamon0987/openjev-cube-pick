# 実機のセットアップと実行（OMX-F + カメラ 2 台）

音声で指示する前に、ここの手順でアームとカメラを動く状態にします。音声まわりは [voice.md](voice.md) を参照してください。

## 仕組み

```
uv run python -m dlb.voice.agent / dlb twotier --robot real
   │
   ├─ RealOMX (dlb/real/omx.py): MuJoCo モデルをデジタルツインとして持つ
   │    IK はツイン上で解いて関節目標を送り、計測した関節角をツインに写す（tcp・緑十字・画像の向きはここから計算）
   │        │ rclpy: /leader/joint_trajectory（腕 5 関節 + gripper_joint_1 をまとめて送る）
   │        ▼
   │   open_manipulator_bringup omx_f_follower_ai.launch.py（JointTrajectoryController, /dev/ttyACM0）
   │
   ├─ 俯瞰カメラ（設定名 front）: VLM がグリッパー・物体・ビンに点を付け、机上→画素の変換をその場で推定
   │    → キューブ / ビンの上まで大まかに移動（カメラ較正もマーカーも不要）
   └─ 手首カメラ（設定名 wrist）: グリッパー真下に緑十字を描き、判断層（openjev）に「対象は十字の左か右か」
        「上か下か」を聞いて位置を合わせる → 真下に降りて掴む
```

物体の座標を判断層に渡すことはありません。必要な幾何は「ロボット自身の関節角」と「手首カメラのモデル」だけです。

## 1. ROS 2 と OMX-F のドライバ

Ubuntu 24.04 + ROS 2 Jazzy を前提にします。

1. `~/ros2_ws/src` に ROBOTIS の `open_manipulator`（main）、`cyclo_control`、`robotis_interfaces`、`dynamixel_hardware_interface` を取得し、`rosdep install --from-paths src -y --ignore-src` → `colcon build`。
2. シリアルポートを使えるようにする: `sudo usermod -aG dialout $USER`（再ログインが必要）。

## 2. Python 環境

`rclpy` は PyPI からは入らないので、ROS 2 の Python が見える仮想環境を作ります。

```bash
source /opt/ros/jazzy/setup.bash && source ~/ros2_ws/install/setup.bash
uv venv --system-site-packages -p /usr/bin/python3
uv pip install -e ".[real,voice,dev]"   # numpy は pyproject で 1.x に固定（ROS 2 Jazzy に合わせる）
```

以降のコマンドは、この 2 つの `source` をしたシェルで実行します。

## 3. アームの起動と動作確認

`configs/robot/omx_f.yaml` の `ros_domain_id`（既定 42）と、launch 側の `ROS_DOMAIN_ID` をそろえます。

```bash
# ターミナル 1: フォロワーアームのドライバ（U2D2 を接続して電源 ON）
export ROS_DOMAIN_ID=42
ros2 launch open_manipulator_bringup omx_f_follower_ai.launch.py port_name:=/dev/ttyACM0

# ターミナル 2
uv run python scripts/real_check.py            # 読むだけ: 関節角・ツインの tcp・グリッパーの読み値
uv run python scripts/real_check.py --home     # 低速で工具真下向きの姿勢へ
uv run python scripts/real_check.py --jog      # 各方向に 1 cm 動かして戻す
uv run python scripts/real_check.py --gripper  # 開閉。読み値を omx_f.yaml の gripper_open / gripper_closed に反映
```

各動作の前に Enter で確認を求めます。Ctrl+C で止めると、アームは最後の目標を保持します。

ロボットをつながずに ROS 経由の経路だけ試すときは、launch に `use_mock_hardware:=true` を付け、`dlb twotier --robot mock ...` で動かします（カメラ画像はツインの仮想シーンを描画したもの）。

## 4. 開始姿勢と机の高さを登録する

`omx_f.yaml` の `poses` と `z_floor_m` は筆者のリグの値です。自分のリグで取り直してください。

```bash
uv run python scripts/real_poses.py --capture begin   # トルク OFF → 手でアームを動かして Enter
uv run python scripts/real_poses.py --capture rest    # 待機用のたたんだ姿勢
uv run python scripts/real_poses.py --z-floor         # トルク OFF → 指先を机に付けて Enter
uv run python scripts/real_poses.py --list            # 登録した姿勢と関節可動域の余裕を表示
```

`begin` は、作業領域の上でグリッパーを下向きに傾けた姿勢にします（手首カメラが机をほぼ真下に見る角度。関節の可動域に 8° 以上の余裕を残す）。以後の IK はこの姿勢の傾きを保ちます。

## 5. カメラ

1. `v4l2-ctl --list-devices` でデバイス番号を調べ、`omx_f.yaml` の `cameras.wrist.device` / `cameras.front.device` に書く（`/dev/videoN` の N）。
2. **俯瞰カメラ（front）** は机を真上から見下ろし、ロボットの前方が画像の上になる向きに置く。較正は不要。キューブは 3 cm 角を想定しています（見かけの大きさを縮尺の初期値に使う）。
3. **手首カメラ（wrist）** はグリッパーに固定し、次の較正を 1 回行う。カメラの取り付けや `begin` 姿勢を変えたらやり直します。

```bash
# begin 姿勢で手首カメラに写る位置にキューブを置いてから
uv run python scripts/calibrate_wrist_model.py
```

アームが `begin` の周りを格子状に動き、各姿勢でキューブの色の塊を検出します。そこから手首カメラの内部パラメータとグリッパーへの取り付け位置を推定し、`configs/robot/calib_wrist.yaml` に書き込みます。既定の色の範囲はオレンジ系です。別の色のキューブを使うときは `--hsv-lo` / `--hsv-hi`（OpenCV の HSV）で指定してください。リポジトリに入っている `calib_wrist.yaml` は筆者のリグのものなので、必ず上書きしてください。

## 6. 判断層（openjev）を立てる

[openjev.md](openjev.md) の手順で起動し、`uv run dlb smoke --backend openjev` が通ることを確認します。初回はモデルのダウンロードとカーネルのビルドで 20〜30 分かかります。

## 7. 動かす

まず掴むところまでを試します（キューブを掴んだ状態で止まる）。

```bash
uv run python scripts/real_grasp_only.py --object-names "orange cube,black bin"
```

キューブを掴んでビンに入れるまでを 1 回通すときは次のコマンドです。`--object-names` には、俯瞰画像で VLM が見分けられる名前を「キューブ,ビン」の順に書きます。

```bash
uv run dlb twotier --backend openjev --robot real --policy bisect --planner sequence --overhead-guide \
  --cameras wrist --object-names "orange cube,black bin" --episodes 1
```

俯瞰画像へのマーキングは OpenAI API（既定 gpt-5.5、5 並列）を呼ぶので、`.env` に `OPENAI_API_KEY` が必要です。

音声で指示するときは [voice.md](voice.md) に進んでください。

## 安全

- 速度は `omx_f.yaml` の `max_tcp_speed` / `fast_tcp_speed` で制限しています。最初は小さめの値で試してください。
- 作業範囲の外の目標と、関節が `max_joint_step` 以上跳ぶ IK 解は送りません。目標に届かない移動は失敗として止まります。
- Ctrl+C で止めるとアームは最後の姿勢を保持します。非常時は電源を切ってください（トルクが抜けてアームが落ちるので、下に手や物を置かないこと）。
- 掴み損ねや失敗で止まったとき、何も握っていなければ開始姿勢に戻ります。握っているときはその場で止まります。
