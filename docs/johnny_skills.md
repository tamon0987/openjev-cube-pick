# Johnny のスキル

Johnny（`dlb/johnny/`）は、ウェイクワードのあとの発話を LLM（gpt-5.5、ツール呼び出し）が受け取り、リグへの操作を
**スキル**として呼び出す音声エージェントです。このページはスキル側（`dlb/johnny/skills/*.py`）の説明です。
インターフェースは `dlb/johnny/skills/base.py`（`Skill` の `name` / `description` / `parameters` / `run(ctx, **kwargs)` /
`cancel(ctx)`、共有状態の `Context`）で、レジストリが `dlb.johnny.skills` 配下の全モジュールを読み込み、
`Skill` のサブクラスを引数なしで生成します。

```
発話 ─▶ LLM ─▶ スキル（run）─▶ SkillResult.text（LLM に返る）─▶ ユーザーへの返事
                              └─ ctx.say(...)：動く前の注意など、LLM を介さずすぐ伝える
```

| スキル | 引数 | すること |
|---|---|---|
| `rig_status` | `deep` | 見るだけ。カメラ 2 台（画像が来るか）、アームの USB ポート、フォロワーのドライバ、openjev、文字起こしサーバー、`OPENAI_API_KEY`。`deep=true` で `dlb smoke --backend openjev` も実行 |
| `rig_setup` | `start`（`follower` / `openjev` / `stt` の配列、既定は全部）、`robot`（既定 true） | 確認して、足りないものを起動し、サービスがそろっていれば**ロボットも起動する**（はじめの姿勢 + 約 22 秒の俯瞰カメラ自己校正。アームが動く前に `ctx.say` で知らせる）。openjev がまだ起動中なら、その旨を返し、次の `rig_setup` / `calibrate_overhead` / `pick_place` が起動する |
| `rig_shutdown` | `docker` | `safe_stop` と同じ手順（下記）。`docker=true` で openjev と stt のコンテナも停止 |
| `identify_cameras` | `step`, `rotate` | どの USB カメラが手首/俯瞰かを、ユーザーに 1 台ずつ抜いてもらって特定し、設定に書く（対話式、下記） |
| `identify_arms` | `step` | どのポートがフォロワー/リーダーかを、リーダーの USB を抜いてもらって特定（対話式） |
| `teleop_start` / `teleop_stop` / `teleop_status` | なし | リーダーアーム（OMX-L）でフォロワーを動かす |
| `pick_place` | `instruction`, `objects`, `place`, `flat_place` | 発話をそのまま渡す。ロボットが未起動なら起動し、`dlb.voice.agent.Agent` で解釈・実行。把持は手首カメラで、置けたかは俯瞰画像で**確認してから**報告する（下記） |
| `robot_home` / `robot_release` | なし | はじめの姿勢へ / グリッパーを開く（ロボット未起動なら起動する） |
| `calibrate_overhead` | なし | 俯瞰カメラの自己校正をやり直す（カメラを動かした・替えたとき） |
| `stop_all` | `word`（任意） | **即時停止**（「止まって」「ストップ」）: 実行中のタスクをキャンセルし、テレオペを止める。アームはその場で保持。何も動いていないときに `word` が「停止」「終了」なら `safe_stop` を実行 |
| `safe_stop` | `docker` | **計画的な停止**（「停止して」「終了して」「今日は終わり」）: タスク中止 → テレオペ停止 → グリッパーを開く → 休止姿勢へゆっくり → ロボット解放 → フォロワーのドライバ停止（アームが休んだ状態でトルクが切れる） |
| `status` | なし | 1 行の現状（安い） |

結果の `text` は LLM 向けの短い英語、ユーザーに手を動かしてもらう指示（`data.ask_user = true`）は日本語で、LLM がそのまま伝えます。

## セットアップの確認（`rig_status` / `rig_setup`）

`rig_setup` は、役割が未確認のときに次の対話を行い、完了後にサービスとロボットを起動します。

1. 「フォロワーアームのUSBだけを抜いてください」→「外した」
2. 消えたUSB固有IDをフォロワー、残った1台をリーダーとして確認し、「フォロワーを挿し直してください」
3. 再接続を確認したら、「フォロワーアームの手首カメラのUSBを抜いてください」→「外した」
4. 手首カメラを確認し、残った外付けカメラを俯瞰カメラとして案内して、再接続を待ちます。
   外付けカメラが3台以上なら、俯瞰カメラも抜き差しして確認します。
5. 保存した比較画像で俯瞰カメラの回転を確認してから、通常のセットアップに進みます。

各返答後は同じ `rig_setup` を `step=data.next` で呼びます。最初の `start` / `robot` オプションは対話中保持します。
`robot=false` ならカメラ識別とロボット起動を省略し、アームの確認だけ行います。
確認済みのUSB固有IDとカメラ設定が現在の接続に一致すれば、抜き差しは省略します。ポート番号の変更は固有IDで追跡します。
アームが1台しか見つからなくてもフォロワーとは決めつけません。ドライバやテレオペが起動中なら、未確認アームのUSBを
抜くようには案内せず、先に終了を求めます。`step=cancel` や「止まって」で中断できます。途中までに確認・保存した設定は残ります。

確認する項目と、足りないときの扱い:

- **カメラ**: `configs/robot/omx_f.yaml` の `cameras.wrist.device` / `cameras.front.device`（`/dev/v4l/by-id/` のパス）が
  つながっていて、OpenCV で開いて 1 フレーム取れるか（取ったら解放）。ノート PC 内蔵カメラ（名前に `Integrated_Webcam`）は
  数えません。設定にない → `identify_cameras` を勧める。画像が来ない → 「手首カメラから画像が来ません。USB を挿し直して
  ください」と `ask_user` で返し、次の呼び出しで確認し直す。
- **アーム**: `/dev/serial/by-id/*OpenRB-150*`。確認済みならその by-id を今の `/dev/ttyACMn` に解決します。
  `rig_setup` では未特定のまま起動せず、上の対話へ進みます。読み取りだけの `rig_status` は未特定なら
  ポート番号からの仮定である旨を表示し、`identify_arms` を勧めます。
- **フォロワーのドライバ**: 自分で起動した launch が生きているか、起動済みのロボットに `/joint_states` が届いているか、
  `ros2 topic list` に `/joint_states` があるか。起動は
  `ros2 launch open_manipulator_bringup omx_f_follower_ai.launch.py port_name:=/dev/ttyACM0`
  を `bash -lc 'source /opt/ros/jazzy/setup.bash && source ~/ros2_ws/install/setup.bash && exec ...'` で、
  設定の `ros_domain_id`（42）を `ROS_DOMAIN_ID` に入れ、新しいセッションで起動。ログは `results/johnny/follower.log`、
  `ctx.processes["follower"]` に登録。`/joint_states` が出るまで最大 25 秒待つ。
- **openjev**: `http://127.0.0.1:8080` の `/health`（`/ready`、`/v1/models` も可）。コンテナがあれば `docker start openjev`、
  なければ `HF_HUB_OFFLINE=1 bash scripts/serve_openjev.sh`。健康になるまで 1〜3 分かかるので「起動中」とだけ返し、
  あとで `rig_status` で確認する。
- **文字起こし**: `http://127.0.0.1:8010/health`。`docker start stt` か `bash scripts/stt_server.sh`。
- **`OPENAI_API_KEY`**: `.env` か環境変数。

俯瞰カメラを `identify_cameras` で替えた・回したあとは `calibrate_overhead` を勧めます（ロボット起動時にも校正します）。

## 対話式の特定（`identify_cameras` / `identify_arms`）

ステップマシンです。状態は `ctx.state["identify_cameras"]` / `ctx.state["identify_arms"]` にあり、各呼び出しは
1 ステップだけ進めて、ユーザーへの指示を `text` に、`data = {"ask_user": true, "next": "<次のステップ>"}` を返します。
LLM が指示を伝え、ユーザーが「抜いた」「挿した」と言ったら `step=<next>` で呼び直します。何も変わっていなければ
同じ指示を繰り返すだけで、時間が経っても問題ありません。`step="cancel"` で中断。

`identify_cameras`:

1. `start`: つながっているカメラの一覧を控え、「手首カメラの USB を抜いてください」
2. `wrist_unplugged`: 消えた 1 台が手首カメラ。残りが 1 台ならそれが俯瞰カメラ（→ `plugged`）、複数なら
   「手首を挿し直して、俯瞰カメラを抜いてください」（→ `front_unplugged`）
3. `plugged`: 両方つながっていて画像が来ることを確認し、`cameras.wrist.device` / `cameras.front.device` を
   `scripts/setup_cameras.py` の `write_config` で書く（コメントは保たれる）。俯瞰画像を 0/90/180/270 度回して並べた
   `results/johnny/cameras_front.jpg` を保存し、「ロボットが下側に来るのはどの角度ですか？（今は 180 度）」
4. `rotation`（`rotate=0|90|180|270`）: `cameras.front.rotate` を書き、`results/johnny/rig.json` に控える。
   `ctx.state["cameras_changed"]` を立てて `calibrate_overhead` を勧める。

`identify_arms`: `start`（「リーダーアームの USB を抜いてください」）→ `leader_unplugged`（消えたポートがリーダー、残りが
フォロワー。by-id の名前で `ctx.state["arms"]` と `rig.json` に保存）→ `plugged`（挿し直したあとの `/dev/ttyACMn` を解決）。
設定ファイルには書きません。フォロワーの launch とテレオペは、この結果からポートを取ります。

## テレオペ（`teleop_start` / `teleop_stop` / `teleop_status`）

ROBOTIS の `omx_l_leader_ai.launch.py` は、`/leader` 名前空間の `joint_trajectory_command_broadcaster` がリーダーの
関節を `trajectory_msgs/JointTrajectory`（`time_from_start = 0`、100 Hz）として **`/leader/joint_trajectory`** に出し、
フォロワーの launch（`omx_f_follower_ai.launch.py`）は `ros2_control_node` の `/arm_controller/joint_trajectory` を
その topic にリマップしています。つまりそのまま上げると、**フォロワーがリーダーの姿勢へ全速で飛びます**（このリグで
joint2 が -1.86 rad まで飛び、ID 13 / 15 が過負荷エラーになりました）。グリッパーもリーダーの値（`reverse_joints`
で符号反転した値、閉じ側を越えて -0.29 など）がそのまま届きます。そこで Johnny は直結せず、中継を挟みます。

```
リーダー launch（dlb/johnny/launch/omx_l_leader_relay.launch.py = ROBOTIS の launch を SetRemap で包む）
   └─ /leader/joint_trajectory_raw ─▶ python -m dlb.johnny.teleop_relay ─▶ /leader/joint_trajectory ─▶ フォロワー
                                       ▲ /joint_states（フォロワーの現在値）
```

- `teleop_start` は、フォロワーのドライバ（なければ起動）→ **中継**（`ctx.processes["relay"]`）→ **リーダーの launch**
  （`ros2 launch <repo>/dlb/johnny/launch/omx_l_leader_relay.launch.py port_name:=/dev/ttyACM1
  raw_topic:=/leader/joint_trajectory_raw`、`ctx.processes["leader"]`）の順に起動し、中継の状態ファイル
  `results/johnny/teleop_status.json` が `following` になるまで（最大 40 秒）待ちます。
- 中継（`dlb/johnny/teleop_relay.py`、ROS を使わない `Relay` クラス + rclpy の薄い `main`）:
  1. `waiting`: リーダーの姿勢とフォロワーの `/joint_states` の両方が届くまで何も出さない。
  2. `approach`: リーダーの姿勢が `OmxKinematics.within_limits`（ファームウェアの関節制限、余裕 0.02 rad）の外、
     または FK で指先が基準面から 2 cm（`--min-z`）より下なら **拒否**（exit code 3、理由を状態ファイルに書く。
     `teleop_start` は両プロセスを止めて理由を返す）。通れば、関節距離 / 0.4 rad/s（最短 2 秒、`go_pose` と同じ速さ）
     の 1 点の軌道でフォロワーをリーダーの姿勢へ**ゆっくり**動かし、到着まで待つ（`ctx.say` で知らせる）。
  3. `following`: リーダーの各メッセージを 50 Hz（`--rate`）に間引き、前回の指令から各関節 **0.02 rad/メッセージ**
     （`--max-step`、= 最大 1.0 rad/s）までに刻んで送る。関節は制限内にクランプ、グリッパーは設定の
     `[gripper_closed, gripper_open]`（0.0〜0.45）にクランプ。
  4. `paused`: リーダーの stream が 0.5 秒、またはフォロワーの `/joint_states` が 1 秒途切れたら送らない
     （フォロワーは保持）。戻れば続きから（離れた分は速度制限つきで追いつく）。
- `teleop_stop` / `stop_all`: 先に中継を止め（SIGINT、送信が止まる）、次にリーダーの launch のプロセスグループに
  SIGINT（10 秒待って SIGTERM、SIGKILL）。フォロワーは最後の姿勢を保持。
- `teleop_status` と `status` は状態ファイルから「approaching / following（送った指令数、速度制限が効いた回数）/
  paused」を 1 行で出し、中継かリーダーの片方だけ死んでいれば BROKEN と言います。
- フォロワー側のグリッパーの対応づけを launch が変えることはありません（フォロワーの `omx_f.ros2_control.xacro` に
  反転やオフセットの設定はなく、中継が送るのはクランプ済みの値だけ）。10/1 と符号が逆に読めたのは過負荷停止中の
  読み値で、電源を入れ直したあとは正常です。
- `open_manipulator_teleop` パッケージ（`omx_f_teleop` はキーボード操作）や `cyclo_teleoperation` は使いません。
- 実機では未検証です。検証したのは、(a) 包んだ launch を `use_mock_hardware:=true`・隔離した `ROS_DOMAIN_ID` で上げると
  `/leader/joint_trajectory_raw` だけが出て `/leader/joint_trajectory` は出ないこと、(b) 中継を `ros2 topic pub` の
  偽の `/joint_states` と raw stream で動かすと、接近 → 追従（グリッパー -0.29 → 0.0 にクランプ）→ SIGINT で停止
  まで動くこと、の 2 点です。
- **排他**: `RealOMX`（pick_place のブリッジ）も同じ `/leader/joint_trajectory` に出すので、
  - タスク実行中（`ctx.state["task_running"]`）は `teleop_start` を拒否、
  - テレオペ中に `pick_place` / `robot_home` / `robot_release` / `calibrate_overhead` を呼ぶと、先にテレオペを止めてから
    実行し（結果に明記）、アームは `ctx.state["arm_moved_externally"]` によりまず `env.reset()`（はじめの姿勢、
    グリッパー開、マークを取り直し）に戻ります。

## ピック＆プレース（`pick_place`）

1. 前提: openjev が健康（1〜3 分の起動中なら「openjev がまだ起動中です（あと 1〜3 分）」と返して始めない）、
   フォロワーのドライバ（なければ起動）、`OPENAI_API_KEY`、テレオペ停止、他のタスクなし。
2. 初回: `python -m dlb.voice.agent` と同じ引数（`argparse.Namespace`: `text=True, dry_run=False, objects, bin_name,
   flat_place, robot_config, intent_model="gpt-5.5", effort="low", mark_model="gpt-5.5", mark_n=5, remark_s=30,
   max_decisions=40, out="results/johnny", run_name=...`）で `Agent` を作り、`start_robot()` を 1 回だけ呼ぶ
   （はじめの姿勢 → 物体と置き場所の発見（指定がなければ）→ 約 22 秒の自己校正）。`ctx.robot` に保持し、openjev の
   クライアントもこの 1 つだけです。`objects` / `place` / `flat_place` は起動時にしか効きません。
3. `agent.parse(instruction)` → 聞き返し（`clarify`）なら `ok=False, data={"clarify": ..., "ask_user": true}` で
   質問文を返す → `agent.execute(tasks)`（完了までブロック）。
4. **キャンセル**（`cancel(ctx)` / `stop_all`）: ランナーにキャンセルのフックがないので、`runner.max_decisions` と
   `max_planner_calls` を 0 にします。次の判断（または次のプランナー呼び出し）で止まり、**進行中の 1 動作（俯瞰ガイド
   の接近や置く動作の途中なら、その小タスク）は最後まで動きます**。アームはその場で保持。終わったら予算を元に戻します。
   `ctx.state["task_running"]` / `["running_skill"]` が実行中の印です。

## 把持と置いたことの確認（`pick_place`）

- **把持の確認**: グリッパーの開度を使います。把持直後や搬送中に手首画像を Jev へ送る確認は行いません。
  開度だけでは搬送中に物体が滑り落ちても検出できない場合があります。
- **置いたことの確認**（`verify_place`、`dlb/johnny/skills/pick_place.py`）: タスク列が「置く」で終わるとき、
  ランナーがはじめの姿勢に戻ったあとに始まるアイドルマーク（`Agent.idle_mark`）を `guide.join()` で待ち、その
  俯瞰マークで最後にピックした物体の点が置き場所の箱（`Marks.boxes[place]`）の中にあるかを見ます。
  `--flat-place` のときは置き場所のマークから 4 cm 以内（`guide.map` でテーブル座標に直して距離を出す）。
  `here` は物体が見えていれば可。箱がなければ距離で判定。物体が見えなければ「確認できない」。
  使ったマークの代わりに `idle_mark()` をもう一度始めるので、次の指示は新しいマークから始まります。
  結果が「置けていない」なら、`ctx.say`「置けていないので、もう一度やります。」のあと**タスク列を 1 回だけ
  やり直し**（`VERIFY_RETRIES = 1`）、それでも駄目なら `ok=False`、`data.verified = False`、
  `text = "NOT confirmed: ...; placed, but the green cube is NOT in the tray: it lies at (15, 15) cm on the table,
  7 cm from the tray"` で返します（LLM は「完了」と言ってはいけない、と説明文に明記）。エピソードログに
  `verify` レコードを追記します。
- 結果には俯瞰画像による置き場所の確認を書きます。ランナーが途中で止まったとき
  （`stop_reason != planner_complete`）は `NOT done: ...` で `ok=False`。
- 追加の時間: 置いたことの確認はアイドルマーク（gpt-5.5、約 5〜8 秒）を待つ分。
  やり直しが起きたときは 1 タスク分（約 45 秒）が加わります。

## 安全停止（`safe_stop`）と即時停止（`stop_all`）

- `stop_all`（「止まって」「ストップ」、コアがウェイクワードなしの停止語で LLM を介さず呼ぶ）: タスクのキャンセル
  （進行中の 1 動作が終わってから止まる）とテレオペの停止だけ。アームはその場で保持。
- `safe_stop`（「停止して」「終了して」「今日は終わり」「ロボットを止めて」）: 順に (1) 実行中のタスクをキャンセルし、
  終わるまで待つ（最大 90 秒）、(2) テレオペ停止（中継 → リーダー launch）、(3) ロボットが起動していればグリッパーを
  開く（持っていた物はその場に落ちる。アームは高くない）、(4) `env.go_pose("rest")`（設定に `poses.rest` がなければ
  `begin`）へゆっくり、(5) `env.close()`（カメラ解放、`ctx.robot = None`）とフォロワー launch への SIGINT（アームが
  休んだ状態でトルクが切れる）。openjev と STT は `docker=true` のときだけ止めます。各ステップを `ctx.say` で
  伝え、結果の文に順番に並べます。アームの移動に失敗したら、トルクは切らずにそこで止めて `ok=False`。
- 「停止」は現在コアの STOP_WORDS にあり、`stop_all` が LLM なしで呼ばれます。`stop_all` に `word`（聞こえた停止語）
  を渡してもらえば、何も動いていないときの「停止」「終了」は `safe_stop` に回し、動いているときは即時停止のままに
  します（`control.is_safe_word`）。

## 校正（`calibrate_overhead`）

`OverheadGuide.calibrate(env)`（`dlb/harness/marking.py`）をやり直します。ロボット未起動なら起動そのものが校正なので
その結果を報告。起動済みなら、グリッパーが空・テレオペなし・タスクなしを確認し、`ctx.say` で「アームがはじめの姿勢の
周りを動き回ります」と伝えてから、はじめの姿勢へ戻り、`guide.map` を新しい `TableMap` にして `calibrate`。結果は
px/cm、回転、9 姿勢のうち何個が一致したか、秒数、却下された姿勢（「pose 3 at (21.0, 3.0) cm: gripper not found」など）。
`RuntimeError` なら理由と、俯瞰画像・`cameras.front.rotate` を確認するよう勧めます。

## 開発時の確認

```bash
PYTHONPATH=. MUJOCO_GL=egl .venv/bin/python -m pytest -q tests/test_johnny_skills.py   # ROS もカメラも不要（全部フェイク）
.venv/bin/ruff check . && .venv/bin/ruff format --check .
```

`tests/test_johnny_skills.py` は `dlb/johnny/skills/rig.py` の機器アクセス（デバイス一覧、フレーム取得、HTTP、
`subprocess`、launch の起動/停止）をフェイクに差し替えて、各スキルの判断・ステップマシン・排他・キャンセルを確かめます。
実機（アームが動く、launch が上がる、カメラが開く、Docker）は未検証です。
