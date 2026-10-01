# 実機のセットアップと実行（OMX-F + カメラ 2 台）

音声で指示する前に、ここの手順でアームとカメラを動く状態にします。音声まわりは [voice.md](voice.md) を参照してください。

## 仕組み

```
python -m dlb.voice.agent / dlb twotier --robot real
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
/usr/bin/python3 -m venv --system-site-packages .venv   # sudo apt install python3-venv が必要
source .venv/bin/activate
pip install -e ".[real,voice,dev]"   # numpy は pyproject で 1.x に固定（ROS 2 Jazzy に合わせる）
```

以降のコマンドは、この 3 つの `source`（ROS 2、ワークスペース、`.venv`）をしたシェルで、リポジトリのディレクトリから実行します。新しいターミナルを開いたら、毎回この順に `source` し直してください。

## 3. アームの起動と動作確認

`configs/robot/omx_f.yaml` の `ros_domain_id`（既定 42）と、launch 側の `ROS_DOMAIN_ID` をそろえます。

ドライバを launch するとトルクが入り、アームはその場の姿勢を保持します（launch 引数 `init_position:=true` を付けたときだけ、ROBOTIS の初期姿勢へ動きます。このリポジトリでは使いません）。

```bash
# ターミナル 1: フォロワーアームのドライバ（U2D2 を接続して電源 ON）
export ROS_DOMAIN_ID=42
ros2 launch open_manipulator_bringup omx_f_follower_ai.launch.py port_name:=/dev/ttyACM0

# ターミナル 2
python scripts/real_check.py            # 読むだけ: 関節角・ツインの tcp・グリッパーの読み値
python scripts/real_check.py --home     # 低速で工具真下向きの姿勢へ
python scripts/real_check.py --jog      # 各方向に 1 cm 動かして戻す
python scripts/real_check.py --gripper  # 開閉。読み値を omx_f.yaml の gripper_open / gripper_closed に反映
python scripts/real_check.py --held     # 指の間にキューブを持って閉じる。表示された値を gripper_held_margin に反映
```

`gripper_held_margin` は「キューブを握れたか」の判定に使います（閉じた読み値が `gripper_closed` からも `gripper_open` からもこれ以上離れていれば、何かを握っている）。`--gripper` で開閉の値を反映してから `--held` を実行してください。

各動作の前に Enter で確認を求めます。Ctrl+C で止めると、アームは最後の目標を保持します。

ロボットをつながずに ROS 経由の経路だけ試すときは、launch に `use_mock_hardware:=true` を付け、`dlb twotier --robot mock ...` で動かします（カメラ画像はツインの仮想シーンを描画したもの）。

## 4. 開始姿勢と机の高さを登録する

`omx_f.yaml` の `poses` と `z_floor_m` は筆者のリグの値です。`--list` で表示される指先の位置が自分のリグの机・作業領域と合わないとき（アームの取り付けや机の高さが違うとき）は、次の手順で取り直してください。

```bash
python scripts/real_poses.py --capture begin   # トルク OFF → 手でアームを動かして Enter
python scripts/real_poses.py --capture rest    # 待機用のたたんだ姿勢
python scripts/real_poses.py --z-floor         # トルク OFF → 指先を机に付けて Enter
python scripts/real_poses.py --list            # 登録した姿勢と関節可動域の余裕を表示
```

`begin` は、作業領域の上でグリッパーを下向きに傾けた姿勢にします（手首カメラが机をほぼ真下に見る角度。関節の可動域に 8° 以上の余裕を残す）。以後の IK はこの姿勢の傾きを保ちます。

## 5. カメラ

**手首カメラ（wrist）** はグリッパーに固定し、**俯瞰カメラ（front）** は机を真上から見下ろすように置きます。俯瞰カメラは較正不要です。キューブは 3 cm 角を想定しています（俯瞰画像での見かけの大きさを縮尺の初期値に使う）。

### 5.1 どのカメラが手首／俯瞰かを登録する

カメラを付け替えたり挿し直したりしたら、毎回やり直します（`/dev/videoN` の番号は挿し直しや再起動で入れ替わるので、設定には変わらない `/dev/v4l/by-id/` のパスを書きます）。

```bash
python scripts/setup_cameras.py
```

1. 接続中のカメラから 1 枚ずつ撮り、番号付きで `results/cameras.jpg` に並べます。手首カメラ（机を間近に見ている）と俯瞰カメラ（リグ全体が写っている）の番号を入力します。
2. 俯瞰画像を 0 / 90 / 180 / 270° 回したものを `results/cameras_front.jpg` に並べます。**ロボットが画像の下側に来る**角度を入力します（俯瞰画像へのマーキングがこの向きを前提にしています）。
3. 結果は `omx_f.yaml` の `cameras.wrist.device`、`cameras.front.device`、`cameras.front.rotate` に書き込まれます。

### 5.2 手首カメラの較正を確かめる（合わなければ較正する）

リポジトリには筆者のリグで較正した `configs/robot/calib_wrist.yaml` が入っています。まず、それがそのまま使えるかを確かめます（キューブは不要）。

```bash
python scripts/calibrate_wrist_model.py --check
```

アームが `begin` 姿勢でグリッパーを閉じ、較正ファイルから計算した指先の位置に緑の印を描いた手首画像を `results/calib_wrist/<日時>/check.jpg` に保存します。**印が左右の指先の先端の中点（10 px 程度以内）にあれば、較正は不要**です。ずれているとき、手首カメラを付け直したとき、`begin` 姿勢を取り直したときは、次の較正を行います。

```bash
# begin 姿勢で手首カメラの視野の真ん中あたりにキューブを置いてから
python scripts/calibrate_wrist_model.py
```

1. アームが `begin` 姿勢に移動し、キューブが手首カメラの視野の中ほどに見えるかを確かめます。見えないときは、その画像を `results/calib_wrist/<日時>/begin.jpg` に保存して止まるので、置き直して再実行します。
2. グリッパーを閉じ、目盛り付きの手首画像を `results/calib_wrist/<日時>/fingertips.jpg` に保存します。**左右の指先の先端の中点**の画素を、目盛り（320 px 単位）で読んで `u v` の形で入力します。前回の値があれば Enter でそのまま使えます（カメラを付け直したら測り直す）。
3. アームが `begin` の周りを格子状（±2 cm、高さ 2 段）に動き、各姿勢でキューブの色の塊を検出します。そこから手首カメラの焦点距離とグリッパーへの取り付け位置を推定し、`configs/robot/calib_wrist.yaml` に書き込みます。再投影誤差（`reprojection mean`）が 10 px 前後なら問題ありません。

既定の色の範囲はオレンジ系です。別の色のキューブを使うときは `--hsv-lo` / `--hsv-hi`（OpenCV の HSV）で指定してください。

## 6. 判断層（openjev）を立てる

[openjev.md](openjev.md) の手順で起動し、`dlb smoke --backend openjev` が通ることを確認します。初回はモデルのダウンロードとカーネルのビルドで 20〜30 分かかります。

## 7. 動かす

まず掴むところまでを試します（キューブを掴んだ状態で止まる）。この試験は俯瞰画像を使わず手首カメラだけで位置を合わせるので、キューブは `begin` 姿勢で手首カメラに写る位置（指先の 1〜2 cm 先、5.2 の較正で置いたあたり）に置いてください。

```bash
python scripts/real_grasp_only.py --object-names "orange cube,black bin"
```

キューブを掴んでビンに入れるまでを 1 回通すときは次のコマンドです。こちらは俯瞰画像でキューブの上まで移動するので、キューブはアームが届く範囲のどこに置いてもかまいません。`--object-names` には、俯瞰画像で VLM が見分けられる名前を「キューブ,ビン」の順に書きます。

```bash
dlb twotier --backend openjev --robot real --policy bisect --planner sequence --overhead-guide \
  --cameras wrist --object-names "orange cube,black bin" --episodes 1
```

俯瞰画像へのマーキングは OpenAI API（既定 gpt-5.5、5 並列）を呼ぶので、`.env` に `OPENAI_API_KEY` が必要です。

最後に表示される集計の `success_rate` は、デジタルツイン上の仮想キューブで判定するシミュレーション用の値で、実機では常に 0 になります。キューブがビンに入ったかは目で確かめてください（ログは `results/twotier/logs/` に画像付きで残ります）。

音声で指示するときは [voice.md](voice.md) に進んでください。

## 安全

- 速度は `omx_f.yaml` の `max_tcp_speed` / `fast_tcp_speed` で制限しています。最初は小さめの値で試してください。
- 作業範囲の外の目標と、関節が `max_joint_step` 以上跳ぶ IK 解は送りません。目標に届かない移動は失敗として止まります。
- Ctrl+C で止めるとアームは最後の姿勢を保持します。非常時は電源を切ってください（トルクが抜けてアームが落ちるので、下に手や物を置かないこと）。
- 掴み損ねや失敗で止まったとき、何も握っていなければ開始姿勢に戻ります。握っているときはその場で止まります。
