#!/usr/bin/env bash
set -euo pipefail

# Canonical launcher for the live PersonaPlex + IMTalker winners.
# AH is AJ plus anti-burst audio pacing and is the recommended default.

IMTALKER_DIR="${IMTALKER_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PROJECT_ROOT="${PROJECT_ROOT:-$(dirname "$IMTALKER_DIR")}"
VENV_DIR="${VENV_DIR:-/workspace/preprocess_5090}"
VARIANT="${VARIANT:-AH}"
VARIANT="${VARIANT^^}"

pick_existing() {
  local candidate
  for candidate in "$@"; do
    if [[ -e "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  printf 'Missing required asset. Checked:\n' >&2
  printf '  %s\n' "$@" >&2
  return 1
}

case "$VARIANT" in
  AH)
    SERVER_FILE="$IMTALKER_DIR/liveTryHeliumFrontendDequeStaticPoseFP32FM_ws_binary_AHAudioPace.py"
    DEFAULT_PORT=8998
    DEFAULT_GPU=0
    DEFAULT_CFG=1.13
    DUMP_NAME=typeah_audio_pace
    ;;
  AJ)
    SERVER_FILE="$IMTALKER_DIR/liveTryHeliumFrontendDequeStaticPoseFP32FM_ws_binary_AJNetworkIso.py"
    DEFAULT_PORT=8999
    DEFAULT_GPU=1
    DEFAULT_CFG=1.15
    DUMP_NAME=typeaj_network_iso
    ;;
  *)
    echo "VARIANT must be AH or AJ, got: $VARIANT" >&2
    exit 2
    ;;
esac

PORT="${PORT:-$DEFAULT_PORT}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$DEFAULT_GPU}"
A_CFG_SCALE="${A_CFG_SCALE:-$DEFAULT_CFG}"
NFE="${NFE:-5}"
VOICE_PROMPT="${VOICE_PROMPT:-VARM3.pt}"

PERSONAPLEX_DIR="${PERSONAPLEX_DIR:-$(pick_existing \
  /workspace/personaplex_bnb4 \
  "$PROJECT_ROOT/checkpoints/personaplex_bnb4")}"
ADAPTER_PATH="${ADAPTER_PATH:-$(pick_existing \
  /workspace/hf_assets/personaplex_lookahead_rms_adapter/checkpoints/personaplex_lookahead096_future048_rms50_adapter.pt \
  "$PROJECT_ROOT/checkpoints/personaplex_lookahead_rms_adapter/checkpoints/personaplex_lookahead096_future048_rms50_adapter.pt")}"
SILENCE_HELIUM_PATH="${SILENCE_HELIUM_PATH:-$(pick_existing \
  /workspace/hf_assets/personaplex_lookahead_rms_adapter/stats/silence_helium_mean.pt \
  /workspace/personaplex_frontend_adapter_dataset/stats/silence_helium_mean.pt \
  "$PROJECT_ROOT/checkpoints/personaplex_lookahead_rms_adapter/stats/silence_helium_mean.pt")}"
DISABLE_LORA="${DISABLE_LORA:-0}"
if [[ "$DISABLE_LORA" == "1" ]]; then
  LORA_GENERATOR_PATH=""
  LORA_ARGS=()
else
  LORA_GENERATOR_PATH="${LORA_GENERATOR_PATH:-$(pick_existing \
    "$PROJECT_ROOT/checkpoints/live_winner/lora/ditto_blink_lora_withaudio_r64_096_continue_2h_last.ckpt" \
    "$IMTALKER_DIR/checkpoints/ditto_blink_lora_withaudio_r64_1h_last.ckpt" \
    /workspace/hf_assets/lora/ditto_blink_lora_withaudio_r64_1h_last.ckpt \
    "$PROJECT_ROOT/checkpoints/lora/ditto_blink_lora_withaudio_r64_1h_last.ckpt")}"
  LORA_ARGS=(
    --lora_generator_path "$LORA_GENERATOR_PATH"
    --lora_rank 64
    --lora_alpha 128
    --lora_dropout 0.05
  )
fi

# RAG / tool-calling / web-search -- opt-in, additive. ENABLE_RAG=0 (default)
# reproduces the exact pre-RAG launch command with zero new flags appended.
ENABLE_RAG="${ENABLE_RAG:-0}"
if [[ "$ENABLE_RAG" == "1" ]]; then
  RAG_CHECKPOINT_DIR="${RAG_CHECKPOINT_DIR:-$(pick_existing \
    "$PROJECT_ROOT/checkpoints/rag_lora" \
    /workspace/hf_assets/rag_lora)}"
  RAG_INDEX_DIR="${RAG_INDEX_DIR:-$(pick_existing \
    "$PROJECT_ROOT/personaplex/rag_index")}"
  STT_PKG_DIR="${STT_PKG_DIR:-$PROJECT_ROOT/checkpoints/stt}"
  CONVERSATION_LOG_DIR="${CONVERSATION_LOG_DIR:-$PROJECT_ROOT/conversation_logs}"
  THINKING_SOUND_PATH="${THINKING_SOUND_PATH:-$PROJECT_ROOT/personaplex/ai-thinking-sound.wav}"
  RAG_ARGS=(
    --conversation_log_dir "$CONVERSATION_LOG_DIR"
    --rag_checkpoint_dir "$RAG_CHECKPOINT_DIR"
    --rag_index_dir "$RAG_INDEX_DIR"
    --stt_hf_repo "${STT_HF_REPO:-kyutai/stt-1b-en_fr-candle}"
    --stt_pkg_dir "$STT_PKG_DIR"
    --vad_threshold "${VAD_THRESHOLD:-0.5}"
    --compressor_model "${COMPRESSOR_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
    --compressor_device "${COMPRESSOR_DEVICE:-cuda}"
    --thinking_sound_path "$THINKING_SOUND_PATH"
    # Forensic fix (conversation_logs_1/2/3): real retrieval+compression was
    # observed regularly taking 2.5-3.7s end to end, so the old fixed ~2.0s
    # filler cap discarded a correctly-computed answer in every logged RAG
    # turn. Default raised to 6.0s with comfortable margin; override here if
    # your corpus/compressor/network profile needs something different.
    --rag_max_filler_sec "${RAG_MAX_FILLER_SEC:-6.0}"
    # Forensic fix: web results had no relevance floor at all (scores as low
    # as 0.04 for clearly-unrelated pages were still used). Local retrieval
    # already floors at rag_min_score; this applies the same discipline to
    # web results.
    --web_search_min_score "${WEB_SEARCH_MIN_SCORE:-0.15}"
  )
  if [[ "${WEB_SEARCH_ENABLED:-0}" == "1" ]]; then
    RAG_ARGS+=(--web_search_enabled --web_search_api_key "${WEB_SEARCH_API_KEY:?set WEB_SEARCH_API_KEY when WEB_SEARCH_ENABLED=1}")
  fi
  echo "RAG enabled: checkpoint=$RAG_CHECKPOINT_DIR index=$RAG_INDEX_DIR stt_pkg=$STT_PKG_DIR web_search=${WEB_SEARCH_ENABLED:-0} conversation_log_dir=$CONVERSATION_LOG_DIR thinking_sound=$THINKING_SOUND_PATH rag_max_filler_sec=${RAG_MAX_FILLER_SEC:-6.0} web_search_min_score=${WEB_SEARCH_MIN_SCORE:-0.15}"
else
  RAG_ARGS=()
fi

GENERATOR_PATH="${GENERATOR_PATH:-$IMTALKER_DIR/checkpoints/generator.ckpt}"
RENDERER_PATH="${RENDERER_PATH:-$IMTALKER_DIR/checkpoints/renderer.ckpt}"
WAV2VEC_MODEL_PATH="${WAV2VEC_MODEL_PATH:-$IMTALKER_DIR/checkpoints/wav2vec2-base-960h}"
REF_PATH="${REF_PATH:-$IMTALKER_DIR/assets/3robert.jpeg}"
PROMPT_FILE="${PROMPT_FILE:-$IMTALKER_DIR/prompts/RB_Robert_System_Prompt_full.txt}"
HTML_PATH="${HTML_PATH:-$IMTALKER_DIR/static/index_v3_binary_fullscreen_aj_nodrop.html}"

for required in \
  "$SERVER_FILE" "$GENERATOR_PATH" "$RENDERER_PATH" \
  "$ADAPTER_PATH" "$SILENCE_HELIUM_PATH" "$REF_PATH" "$PROMPT_FILE" "$HTML_PATH" \
  "$PERSONAPLEX_DIR/model_bnb_4bit.pt" \
  "$PERSONAPLEX_DIR/tokenizer-e351c8d8-checkpoint125.safetensors" \
  "$PERSONAPLEX_DIR/tokenizer_spm_32k_3.model"; do
  [[ -e "$required" ]] || { echo "Missing required path: $required" >&2; exit 1; }
done

if [[ "$ENABLE_RAG" == "1" ]]; then
  for required in \
    "$RAG_CHECKPOINT_DIR/lora/adapter_config.json" \
    "$RAG_INDEX_DIR/manifest.json"; do
    [[ -e "$required" ]] || { echo "Missing required RAG path: $required (re-run scripts/download_live_assets.sh, or set ENABLE_RAG=0)" >&2; exit 1; }
  done
fi

if [[ -z "${VOICE_PROMPT_DIR:-}" ]]; then
  for candidate in \
    "$PERSONAPLEX_DIR/voices" \
    /workspace/.cache/huggingface/hub/models--nvidia--personaplex-7b-v1/snapshots/*/voices \
    /root/.cache/huggingface/hub/models--nvidia--personaplex-7b-v1/snapshots/*/voices \
    "$HOME"/.cache/huggingface/hub/models--nvidia--personaplex-7b-v1/snapshots/*/voices; do
    if [[ -f "$candidate/$VOICE_PROMPT" ]]; then
      VOICE_PROMPT_DIR="$candidate"
      break
    fi
  done
fi
[[ -f "${VOICE_PROMPT_DIR:-}/$VOICE_PROMPT" ]] || {
  echo "Cannot find $VOICE_PROMPT. Set VOICE_PROMPT_DIR explicitly." >&2
  exit 1
}

source "$VENV_DIR/bin/activate"
cd "$IMTALKER_DIR"

export CUDA_VISIBLE_DEVICES
export PYTHONPATH="$IMTALKER_DIR:$PERSONAPLEX_DIR/moshi:$PERSONAPLEX_DIR:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export IMTALKER_PROMPT_STATE_CACHE="${IMTALKER_PROMPT_STATE_CACHE:-1}"

ROBERT_PROMPT="$(tr '\n' ' ' < "$PROMPT_FILE")"
DUMP_DIR="${DUMP_DIR:-$IMTALKER_DIR/live_dumps_${DUMP_NAME}_${PORT}_varm3}"

echo "Starting $VARIANT on port $PORT, physical GPU $CUDA_VISIBLE_DEVICES"
echo "Voice: $VOICE_PROMPT | CFG: $A_CFG_SCALE | NFE: $NFE"
[[ "$DISABLE_LORA" == "1" ]] && echo "Generator: base checkpoint only (LoRA disabled)"

exec python -u "$SERVER_FILE" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --html_path "$HTML_PATH" \
  --generator_path "$GENERATOR_PATH" \
  --renderer_path "$RENDERER_PATH" \
  "${LORA_ARGS[@]}" \
  --adapter_path "$ADAPTER_PATH" \
  --adapter_type unitalk_last_layer \
  --adapter_num_layers 12 \
  --adapter_dropout 0.0 \
  --adapter_window_mode lookahead \
  --adapter_future_steps 6 \
  --ref_path "$REF_PATH" \
  --wav2vec_model_path "$WAV2VEC_MODEL_PATH" \
  --moshi_root "$PERSONAPLEX_DIR" \
  --mimi_hf_repo nvidia/personaplex-7b-v1 \
  --moshi_weight "$PERSONAPLEX_DIR/model_bnb_4bit.pt" \
  --mimi_weight "$PERSONAPLEX_DIR/tokenizer-e351c8d8-checkpoint125.safetensors" \
  --tokenizer "$PERSONAPLEX_DIR/tokenizer_spm_32k_3.model" \
  --quantize_4bit \
  --text_prompt "$ROBERT_PROMPT" \
  --voice_prompt "$VOICE_PROMPT" \
  --voice_prompt_dir "$VOICE_PROMPT_DIR" \
  --enable_moshi_reply \
  --direct_reply_hidden \
  --reply_hidden_steps_per_chunk 12 \
  --audio_chunk_sec 0.96 \
  --wav2vec_sec 0.96 \
  --fm_chunk_frames 24 \
  --prebuffer_chunks 1 \
  --render_sub_batch 8 \
  --renderer_precision fp32 \
  --frame_q_backpressure 32 \
  --buffer_ms 160 \
  --skip_fm_audio_encoder \
  --assistant_speech_rms_threshold "${ASSISTANT_SPEECH_RMS_THRESHOLD:-0.006}" \
  --assistant_speech_hold_chunks "${ASSISTANT_SPEECH_HOLD_CHUNKS:-1}" \
  --a_cfg_scale "$A_CFG_SCALE" \
  --nfe "$NFE" \
  --seed 42 \
  --noise_seed 42 \
  --shared_noise \
  --fp32 \
  --tf32 \
  --dump_motion \
  --dump_dir "$DUMP_DIR" \
  --silence_helium_path "$SILENCE_HELIUM_PATH" \
  --jpeg_quality 58 \
  --device cuda \
  --reply_audio_gain 1.0 \
  --output_audio_codec opus \
  "${RAG_ARGS[@]}"
