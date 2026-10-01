# 音声で指示する

[real_robot.md](real_robot.md) の手順でアームとカメラが動く状態になっていることが前提です。

```
arecord（既定のマイク）─▶ Silero VAD（発話の区切り）─▶ 文字起こしサーバー :8010 ─▶ 指示の解釈（gpt-5.5）─▶ タスク列 ─▶ ロボット
  dlb/voice/listen.py                                    faster-whisper large-v3-turbo      dlb/voice/intent.py
```

指示は「つかむ（pick）」と「置く（place）」のタスク列に変換されます。置き場所は `bin`（ビンの中）、`here`（その場）、`on:<物体名>`（その物体の上）の 3 種類です。

| 発話の例 | タスク列 |
|---|---|
| 赤いキューブをビンに入れて | pick red cube → place bin |
| それをその場に置いて青いキューブをビンに入れて（何か持っているとき） | place here → pick blue cube → place bin |
| 赤いキューブを青いキューブの上に置いて | pick red cube → place on:blue cube |
| 止まって | （空。キューを消す） |

知らない物体を言われたときなど、指示があいまいなときは実行せずに聞き返します（例:「緑のキューブは見当たりません。赤いキューブか青いキューブのどちらですか？」）。

## 1. 準備

```bash
sudo apt install alsa-utils              # arecord
cp .env.example .env                     # OPENAI_API_KEY を記入（指示の解釈と俯瞰画像のマーキングに使う）
```

マイクは OS の既定の入力デバイスを使います（`arecord -D default`）。別のデバイスを使うときは `arecord -L` で名前を調べて `--device` に渡します（例: `--device plughw:2,0`）。

## 2. 文字起こしサーバーを立てる

```bash
bash scripts/stt_server.sh          # CPU で起動（既定）。http://127.0.0.1:8010/v1（OpenAI 互換）
bash scripts/stt_server.sh cuda     # GPU で起動（openjev と同居させるなら GPU に 1.5 GB 以上の空きが必要）
bash scripts/stt_server.sh stop
```

- [speaches](https://github.com/speaches-ai/speaches)（faster-whisper）の Docker イメージを使います。`--restart unless-stopped` なので、PC を再起動しても自動で上がります。
- 初回に、発話の区切りに使う Silero VAD のモデルを `models/silero_vad.onnx` にダウンロードします。
- openjev が GPU メモリをほぼ使い切るので、既定は CPU です。CPU のスレッド数とコアの割り当ては `STT_CPU_THREADS`（既定 8）と `STT_CPUSET`（既定 `0-7`、空文字で制限なし）で変えられます。発話が終わってから文字になるまで、CPU で約 3 秒かかります。

## 3. 動作確認（ロボットなし）

```bash
python -m dlb.voice.listen                    # マイク → 文字起こしを表示
python -m dlb.voice.listen --intent           # 文字起こしに加えてタスク列も表示
python -m dlb.voice.console                   # キーボードで指示を打ってタスク列を確認
python -m dlb.voice.agent --dry-run --text    # エージェントの解釈だけ（ロボットは動かない）
```

## 4. ロボットを音声で動かす

セットアップが済んでいれば、毎回の起動は次の 3 つです。どのターミナルでも、先に `source /opt/ros/jazzy/setup.bash && source ~/ros2_ws/install/setup.bash` をしておきます。ターミナル 3 ではさらに `source .venv/bin/activate` もします。

```bash
# ターミナル 1: フォロワーアームのドライバ（real_robot.md の 3.）
export ROS_DOMAIN_ID=42
ros2 launch open_manipulator_bringup omx_f_follower_ai.launch.py port_name:=/dev/ttyACM0

# ターミナル 2: openjev と文字起こしサーバー（一度 serve_openjev.sh / stt_server.sh で作ったコンテナを起動）
docker start openjev stt
curl -s localhost:8080/health; curl -s localhost:8010/health   # 両方とも応答が返れば準備完了

# ターミナル 3: 音声エージェント（リポジトリのディレクトリで）
python -m dlb.voice.agent --objects "orange cube,blue cube" --bin-name "black bin"
python -m dlb.voice.agent --text              # 音声の代わりにキーボードで指示する
```

カメラを付け替えたり挿し直したりしたときは、先に `python scripts/setup_cameras.py`（[real_robot.md](real_robot.md) の 5.1）をやり直します。

`ROS_DOMAIN_ID` はドライバ側とエージェント側でそろえます（`configs/robot/omx_f.yaml` の `ros_domain_id`、既定 42）。`~/.bashrc` に `export ROS_DOMAIN_ID=42` を書いておくと楽です。openjev は起動直後の数分 `/health` が返らないことがあるので、返るまで待ってからエージェントを起動してください。

- `--objects` は机の上の物体、`--bin-name` はビンの見た目を、俯瞰画像で VLM が見分けられる名前で書きます。
- 起動するとアームが開始姿勢に移動し、「listening...」と出たら話しかけられます。
- 待機中は俯瞰画像のマーキングを裏で更新しています。人が物体を動かして手を離すと、自動で付け直します。
- 実行中に話した指示は、今のタスクが終わってから実行されます（実行中の割り込みには未対応）。
- 終了は Ctrl+C です。ログは `results/voice/logs/session_<日時>/` に残ります。
