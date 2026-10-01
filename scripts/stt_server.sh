#!/usr/bin/env bash
# Speech-to-text server for voice instructions: speaches (faster-whisper) behind an
# OpenAI-compatible API (POST /v1/audio/transcriptions). Long-lived; the model stays loaded (ttl -1).
#
#   bash scripts/stt_server.sh            # CPU (default: openjev's vLLM holds all but ~0.7 GB of the GPU)
#   bash scripts/stt_server.sh cuda       # GPU, when nvidia-smi shows >= ~1.5 GB free (int8_float16)
#   bash scripts/stt_server.sh stop
#
# Then: python -m dlb.voice.listen    (mic -> transcripts)
#       python -m dlb.voice.listen --file x.wav
set -euo pipefail
MODE="${1:-cpu}"
NAME="${STT_NAME:-stt}"
PORT="${STT_PORT:-8010}"
VERSION="${STT_VERSION:-0.8.3}"               # speaches release, pinned
MODEL="${STT_MODEL:-deepdml/faster-whisper-large-v3-turbo-ct2}"
THREADS="${STT_CPU_THREADS:-8}"               # 8 threads on the 8 P-cores ran as fast as 16 or 24 on all cores
MEM_LIMIT="${STT_MEM_LIMIT:-8g}"
CPUSET="${STT_CPUSET-0-7}"                    # P-cores of the Core Ultra 9 285HX (8-23 are E-cores); "" = any

# client-side VAD model for dlb.voice.listen (Silero v5, MIT); onnxruntime comes with the "voice" extra
VAD="$(dirname "$0")/../models/silero_vad.onnx"
if [ ! -f "$VAD" ]; then
  mkdir -p "$(dirname "$VAD")"
  curl -sL -o "$VAD" https://github.com/snakers4/silero-vad/raw/v5.1.2/src/silero_vad/data/silero_vad.onnx
fi

if [ "$MODE" = "stop" ]; then
  docker rm -f "$NAME"
  exit 0
fi
case "$MODE" in
  cpu)  IMAGE="ghcr.io/speaches-ai/speaches:${VERSION}-cpu";  GPU=();             COMPUTE="${STT_COMPUTE:-int8}" ;;
  cuda) IMAGE="ghcr.io/speaches-ai/speaches:${VERSION}-cuda"; GPU=(--gpus all);  COMPUTE="${STT_COMPUTE:-int8_float16}" ;;
  *) echo "usage: $0 [cpu|cuda|stop]" >&2; exit 2 ;;
esac

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --restart unless-stopped "${GPU[@]}" \
  --memory="$MEM_LIMIT" ${CPUSET:+--cpuset-cpus="$CPUSET"} \
  -p 127.0.0.1:${PORT}:8000 \
  -e WHISPER__COMPUTE_TYPE="$COMPUTE" \
  -e WHISPER__CPU_THREADS="$THREADS" \
  -e WHISPER__TTL=-1 \
  -e ENABLE_UI=false \
  -e LOG_LEVEL=info \
  -v hf-hub-cache:/home/ubuntu/.cache/huggingface/hub \
  "$IMAGE" >/dev/null

echo -n "waiting for http://127.0.0.1:${PORT} "
for _ in $(seq 120); do
  curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null && break
  echo -n .; sleep 1
done
echo
# speaches downloads models on request (no-op once in the hf-hub-cache volume); the Hub drops connections
# now and then, hence the retries. Then one request loads the model so the first utterance is not slow.
for _ in 1 2 3 4 5; do
  curl -sf -X POST "http://127.0.0.1:${PORT}/v1/models/${MODEL}" >/dev/null && break
  sleep 2
done
python3 - "$PORT" "$MODEL" <<'PY'
import io, sys, urllib.request, uuid, wave
buf = io.BytesIO()
with wave.open(buf, "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(bytes(16000))
b = uuid.uuid4().hex
body = (f"--{b}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n{sys.argv[2]}\r\n"
        f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"w.wav\"\r\n"
        f"Content-Type: audio/wav\r\n\r\n").encode() + buf.getvalue() + f"\r\n--{b}--\r\n".encode()
req = urllib.request.Request(f"http://127.0.0.1:{sys.argv[1]}/v1/audio/transcriptions", body,
                             {"Content-Type": f"multipart/form-data; boundary={b}"})
try:
    urllib.request.urlopen(req, timeout=300).read()
except Exception as e:
    print("warm-up failed:", e)
PY
echo "container '$NAME' ($IMAGE) on http://127.0.0.1:${PORT}  model=$MODEL compute=$COMPUTE threads=$THREADS"
echo "  docker logs -f $NAME"
