# 判断層（openjev）のセットアップ

手首カメラの画像を見て「対象は十字の左か右か」に答える判断層として、[openjev](https://github.com/razorback16/openjev) をローカルの GPU で動かします。openjev は TypeSafe Jev と同じ API を持つサーバで、中身は DiffusionGemma 26B-A4B の NVFP4 量子化版です。`scripts/serve_openjev.sh` が Docker で立ち上げます。

## 必要なもの

| | |
|---|---|
| GPU | NVIDIA、VRAM 24 GB 以上。動作確認は RTX PRO 5000 Blackwell Laptop（24 GB, compute capability 12.0）のみ。NVFP4 を使うので Blackwell 世代を想定しています |
| ドライバ | Blackwell 世代なら 570 系以降 |
| Docker | Docker + NVIDIA Container Toolkit（`docker run --gpus all` が通ること） |
| ディスク | 約 30 GB（Docker イメージ 約 6 GB、モデルの重み 約 19 GB、カーネルのビルドキャッシュ） |
| ホストの RAM | 64 GB を推奨（初回のカーネルビルドで数十 GB 使う。下の「メモリが少ないとき」を参照） |

ドライバと Docker の入れ方は [setup_laptop.md](setup_laptop.md) にあります。`bash scripts/check_env.sh` で GPU・Docker・描画の設定をまとめて確認できます。

## 起動

```bash
bash scripts/serve_openjev.sh          # コンテナ名 openjev、http://127.0.0.1:8080
docker logs -f openjev                 # 起動の進み具合を見る
```

初回は次の順に進み、使えるようになるまで 20〜30 分かかります。

1. Docker イメージ `razorback16/openjev:0.5.0` の取得
2. Hugging Face からモデルの重み（`nvidia/diffusiongemma-26B-A4B-it-NVFP4`）を `~/.cache/huggingface` にダウンロード。アクセス制限のないモデルなので、`HF_TOKEN` はなくても取得できます
3. GPU 向けカーネル（FlashInfer）のビルド。約 12 分。結果は `~/.cache/openjev/` に残るので、2 回目以降は行いません

2 回目以降の起動は 2〜3 分です。起動が終わったら、次のコマンドで応答を確認します。

```bash
curl -s http://127.0.0.1:8080/health       # 200 が返れば起動完了
uv run dlb health --backend openjev
uv run dlb smoke  --backend openjev        # テキスト 3 問 + 画像 1 枚。画像の問いで p(red square) が 1 に近ければ OK
```

## 止める・再起動する

```bash
docker stop openjev                        # 止める
docker start openjev                       # もう一度起動（PC の再起動後も自動では上がりません）
docker rm -f openjev                       # コンテナを消す（設定を変えて serve_openjev.sh をやり直すとき）
```

重みとビルドキャッシュはホスト側（`~/.cache/huggingface`、`~/.cache/openjev`）にあるので、コンテナを消しても再ダウンロードはしません。重みを取得したあとは `HF_HUB_OFFLINE=1 bash scripts/serve_openjev.sh` とすると、Hugging Face への接続が不安定なときも起動に失敗しにくくなります。

## 設定（環境変数）

| 変数 | 既定 | 内容 |
|---|---|---|
| `OPENJEV_IMAGE` | `razorback16/openjev:0.5.0` | Docker イメージ。画像入力は 0.2.0 以降で対応（0.1.0 は画像を黙って無視する） |
| `OPENJEV_MAX_LEN` | 8192 | 最大コンテキスト長。上流の既定 65536 は 24 GB には大きすぎる |
| `OPENJEV_GPU_UTIL` | 0.92 | GPU メモリのうち openjev が確保する割合 |
| `OPENJEV_MAX_NUM_SEQS` | 4 | 同時に処理するリクエスト数。16 では起動時に GPU メモリが足りなくなった |
| `OPENJEV_MAX_IMAGES` | 2 | 1 リクエストあたりの画像の枚数 |
| `OPENJEV_JIT_JOBS` | 4 | カーネルビルドの並列数 |
| `OPENJEV_MEM_LIMIT` | 40g | コンテナが使えるホスト RAM の上限 |
| `OPENJEV_PORT` | 8080 | ポート |

## うまく動かないとき

- **GPU メモリが足りない**: openjev は 24 GB のうち約 23 GB を使います。画面表示を内蔵 GPU に任せ、NVIDIA の GPU を計算専用にしてください（`nvidia-smi` で Xorg / gnome-shell が載っていないことを確認）。ブラウザなど GPU を使うアプリも閉じておくと安全です。
- **メモリが少ないとき（ホスト RAM 64 GB 未満）**: 初回のカーネルビルドは 1 プロセスあたり数 GB を使います。並列数を制限しないとホストがメモリ不足で固まることがあります。`OPENJEV_JIT_JOBS=2 OPENJEV_MEM_LIMIT=24g bash scripts/serve_openjev.sh` のように並列数と上限を下げてください（ビルドは遅くなります）。
- **ダウンロード中に切断された**: `docker rm -f openjev` してからもう一度 `serve_openjev.sh`。ダウンロード済みの分はキャッシュに残ります。
- **文字起こしサーバーも GPU で動かしたい**: `OPENJEV_GPU_UTIL=0.86` 程度に下げると約 1.5 GB 空きます（[voice.md](voice.md)）。
