# 判断契約（Decision Contract）

すべてのバックエンドは同じリクエスト／レスポンス形式（TypeSafe Jev の System-One ワイヤ形式）で話します。djev と openjev も同形式で、違いはエンドポイントと画像の有無だけです。

## リクエスト

```json
{
  "model": "jev-latest",                 // TypeSafe / openjev のみ。djev は送らない
  "state": { ... 状態JSON または 文字列 ... },
  "questions": {
    "next_action": {"type": "choice", "instructions": "...", "criteria": {"hover_object": "...", "descend": "...", ...}},
    "holding":     {"type": "noul",   "instructions": "...", "criteria": {"true": "...", "false": "..."}},
    "progress":    {"type": "score",  "instructions": "...", "criteria": ["level0", "level1", ...]}
  },
  "images": ["data:image/png;base64,...", ...]   // 画像対応サーバのみ（openjev / djev）
}
```

制限（2026-09 時点の公開情報）:

| | TypeSafe Jev | djev | openjev |
|---|---|---|---|
| endpoint | `/v1/systemone` | `/v1/request` | `/v1/systemone` |
| 画像 | 不可 | `images`（data URL、≤6 枚、≤5 MiB/枚、≤2048 px、ボディ ≤8 MiB） | README では ≤8 枚・約 280 トークン/枚（フィールド名は要確認） |
| 入力上限 | ~64k トークン（1 質問+state で ~32k） | `DJEV_MAX_MODEL_LEN`（既定 32768） | `--max-model-len`（既定 65536） |
| choice 選択肢 | ≤255 | – | – |
| score 段階 | 2〜10 | – | – |
| 単価 | $0.042 / 1M 入力トークン、出力無料 | 自前 | 自前 |

## レスポンス

```json
{
  "model": "jev-1.13.0",
  "answers": {
    "next_action": {"type": "choice", "choice": "descend", "probabilities": {"descend": 0.84, ...}, "confidence": 0.6},
    "holding":     {"type": "noul", "noul": 0.07},
    "progress":    {"type": "score", "score": 1.3, "confidence": 0.54, "legend": {"0": "...", ...}, "probabilities": {"0": 0.0, "1": 0.7, "2": 0.3}}
  },
  "usage": {"input_tokens": 210, "output_tokens": 31}
}
```

`score.score` は期待値（最頻段階ではない）。本リポの `Answer.label` は score では最大確率の段階を採用します。

## ゲーティングに使う confidence

`Answer.conf` は型に依らず [0,1] に正規化した値です: choice / score は返された `confidence`（なければ top-1 確率）、noul は `|p − 0.5| × 2`。`dlb online --gate τ --on-low {act,oracle,home,stop}` はこの値で分岐します。

## 本リポでの質問セット

- `action_only`: `next_action` のみ（オンライン制御用）。
- `full`: `next_action` + `holding` + `aligned` + `task_complete` + `progress`（オフライン評価で各サブスキルを分離して測る）。

正解は `PickPlaceEnv.oracle_labels()` が特権状態から返します。`state_json` にはこれらの答えを直接含めません（held 等のフラグは入れない）。
