#!/bin/bash
# ============================================================
# Provisioning script per runpod/comfyui:cuda13.0
# ============================================================
# ComfyUI è già installato dal template.
# Questo script installa i custom nodes e scarica i modelli.
# ============================================================

set -e

# Trova dove il template ha installato ComfyUI
if   [ -d "/workspace/ComfyUI" ];    then COMFYUI_DIR="/workspace/ComfyUI"
elif [ -d "/workspace/comfyui" ];    then COMFYUI_DIR="/workspace/comfyui"
elif [ -d "/comfyui" ];              then COMFYUI_DIR="/comfyui"
else
    echo "[ERROR] ComfyUI non trovato. Controlla il template."
    exit 1
fi

MODELS_DIR="$COMFYUI_DIR/models"
NODES_DIR="$COMFYUI_DIR/custom_nodes"
WORKFLOW_DIR="$COMFYUI_DIR/user/default/workflows"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log()  { echo -e "${GREEN}[PROVISION]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }

log "ComfyUI trovato in: $COMFYUI_DIR"
mkdir -p "$NODES_DIR" "$MODELS_DIR" "$WORKFLOW_DIR"

# ────────────────────────────────────────────────────────────
# 1. Dipendenze Python
# ────────────────────────────────────────────────────────────
log "Installando dipendenze Python..."
pip install -q sageattention          || warn "sageattention non installato"
pip install -q flash-attn --no-build-isolation || warn "flash-attn non installato"
pip install -q onnxruntime-gpu

# ────────────────────────────────────────────────────────────
# 2. Custom nodes
# ────────────────────────────────────────────────────────────
install_node() {
    local name="$1"
    local url="$2"
    if [ ! -d "$NODES_DIR/$name" ]; then
        log "Installando $name..."
        git clone --depth 1 "$url" "$NODES_DIR/$name"
    else
        log "Aggiornando $name..."
        cd "$NODES_DIR/$name" && git pull -q
    fi
    [ -f "$NODES_DIR/$name/requirements.txt" ] && \
        pip install -q -r "$NODES_DIR/$name/requirements.txt"
    cd /
}

install_node "ComfyUI-WanVideoWrapper"            "https://github.com/kijai/ComfyUI-WanVideoWrapper"
install_node "ComfyUI-WanAnimatePreprocess"       "https://github.com/kijai/ComfyUI-WanAnimatePreprocess"
install_node "ComfyUI-WanAnimatePreprocess-track" "https://github.com/Herony82/ComfyUI-WanAnimatePreprocess-track"
install_node "ComfyUI-KJNodes"                    "https://github.com/kijai/ComfyUI-KJNodes"
install_node "ComfyUI-VideoHelperSuite"           "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite"
install_node "ComfyUI-SuperNodes"                 "https://github.com/SuperComfy/ComfyUI-SuperNodes"
install_node "cg-use-everywhere"                  "https://github.com/chrisgoringe/cg-use-everywhere"
install_node "rgthree-comfy"                      "https://github.com/rgthree/rgthree-comfy"
install_node "ComfyUI-Easy-Use"                   "https://github.com/yolain/ComfyUI-Easy-Use"
install_node "ComfyUI-compare-videos"             "https://github.com/surinder83singh/ComfyUI-compare-videos"

# ────────────────────────────────────────────────────────────
# 3. Download modelli (skip se già presenti)
# ────────────────────────────────────────────────────────────
download_model() {
    local dest="$1"
    local url="$2"
    local filename
    filename=$(basename "$url" | cut -d'?' -f1)
    mkdir -p "$dest"
    if [ ! -f "$dest/$filename" ]; then
        log "Scaricando $filename..."
        wget -q --show-progress -O "$dest/$filename" "$url"
    else
        log "Già presente: $filename"
    fi
}

# Diffusion model (~28 GB)
download_model "$MODELS_DIR/diffusion_models" \
    "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_animate_14B_bf16.safetensors"

# VAE
download_model "$MODELS_DIR/vae" \
    "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_bf16.safetensors"

# Text encoder UMT5 (~10 GB)
download_model "$MODELS_DIR/text_encoders" \
    "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-bf16.safetensors"

# LoRA Relight
download_model "$MODELS_DIR/loras" \
    "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/LoRAs/Wan22_relight/WanAnimate_relight_lora_fp16.safetensors"

# LoRA LightX2V
# NOTA: aggiorna WanVideoLoraSelectMulti nel workflow col filename corretto
download_model "$MODELS_DIR/loras" \
    "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors"

# ONNX detection
download_model "$MODELS_DIR/detection" \
    "https://huggingface.co/Wan-AI/Wan2.2-Animate-14B/resolve/main/process_checkpoint/det/yolov10m.onnx"

download_model "$MODELS_DIR/detection" \
    "https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/wholebody/vitpose-l-wholebody.onnx"

# ────────────────────────────────────────────────────────────
# 4. Workflow di esempio
# ────────────────────────────────────────────────────────────
if [ ! -f "$WORKFLOW_DIR/Wan_2_2_Animate_tracker_Head_Swap_v01.json" ]; then
    log "Copiando workflow..."
    wget -q -O "$WORKFLOW_DIR/Wan_2_2_Animate_tracker_Head_Swap_v01.json" \
        "https://raw.githubusercontent.com/Herony82/ComfyUI-WanAnimatePreprocess-track/main/workflows/Wan_2_2_-_Animate__tracker__-_Head_Swap_-_v01.json"
fi

log "Provisioning completato. ComfyUI è avviato dal template sulla porta 8188."
