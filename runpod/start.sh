#!/bin/bash
# ============================================================
# RunPod startup script — ComfyUI Wan2.2 Animate Head Swap
# ============================================================

set -e

WORKSPACE="/workspace"
COMFYUI_DIR="$WORKSPACE/ComfyUI"
MODELS_DIR="$COMFYUI_DIR/models"
NODES_DIR="$COMFYUI_DIR/custom_nodes"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log()  { echo -e "${GREEN}[START]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }

# ────────────────────────────────────────────────────────────
# 1. ComfyUI
# ────────────────────────────────────────────────────────────
if [ ! -d "$COMFYUI_DIR" ]; then
    log "Clonando ComfyUI..."
    git clone https://github.com/comfyanonymous/ComfyUI.git "$COMFYUI_DIR"
    pip install -q -r "$COMFYUI_DIR/requirements.txt"
else
    log "ComfyUI trovato, aggiorno..."
    cd "$COMFYUI_DIR" && git pull -q
fi

mkdir -p "$NODES_DIR" "$MODELS_DIR"

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
}

# Pacchetti necessari per il workflow
install_node "ComfyUI-WanVideoWrapper"            "https://github.com/kijai/ComfyUI-WanVideoWrapper"
install_node "ComfyUI-WanAnimatePreprocess"       "https://github.com/kijai/ComfyUI-WanAnimatePreprocess"
install_node "ComfyUI-WanAnimatePreprocess-track" "https://github.com/Herony82/ComfyUI-WanAnimatePreprocess-track"
install_node "ComfyUI-KJNodes"                    "https://github.com/kijai/ComfyUI-KJNodes"
install_node "ComfyUI-VideoHelperSuite"           "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite"
install_node "ComfyUI-SuperNodes"                 "https://github.com/SuperComfy/ComfyUI-SuperNodes"
install_node "cg-use-everywhere"                  "https://github.com/chrisgoringe/cg-use-everywhere"
install_node "rgthree-comfy"                      "https://github.com/rgthree/rgthree-comfy"
install_node "ComfyUI-Easy-Use"                   "https://github.com/yolain/ComfyUI-Easy-Use"
install_node "ComfyUI-Compare-Videos"             "https://github.com/surinder83singh/ComfyUI-compare-videos"

# ────────────────────────────────────────────────────────────
# 3. Dipendenze Python aggiuntive
# ────────────────────────────────────────────────────────────
log "Installando dipendenze Python..."
pip install -q sageattention
pip install -q onnxruntime-gpu  # per i modelli ONNX detection su GPU

# ────────────────────────────────────────────────────────────
# 4. Download modelli (skip se già presenti)
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

# VAE (~1 GB)
download_model "$MODELS_DIR/vae" \
    "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_bf16.safetensors"

# Text encoder UMT5 (~10 GB)
download_model "$MODELS_DIR/text_encoders" \
    "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-bf16.safetensors"

# LoRA — Relight
download_model "$MODELS_DIR/loras" \
    "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/LoRAs/Wan22_relight/WanAnimate_relight_lora_fp16.safetensors"

# LoRA — LightX2V
# NOTA: filename scaricato = rank128, workflow usa rank256
# → aggiorna WanVideoLoraSelectMulti nel workflow col nome corretto
download_model "$MODELS_DIR/loras" \
    "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors"

# ONNX detection → models/detection/
download_model "$MODELS_DIR/detection" \
    "https://huggingface.co/Wan-AI/Wan2.2-Animate-14B/resolve/main/process_checkpoint/det/yolov10m.onnx"

download_model "$MODELS_DIR/detection" \
    "https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/wholebody/vitpose-l-wholebody.onnx"

# ────────────────────────────────────────────────────────────
# 5. Workflow di esempio
# ────────────────────────────────────────────────────────────
WORKFLOW_DIR="$COMFYUI_DIR/user/default/workflows"
mkdir -p "$WORKFLOW_DIR"
if [ ! -f "$WORKFLOW_DIR/Wan_2_2_Animate_tracker_Head_Swap_v01.json" ]; then
    log "Copiando workflow di esempio..."
    wget -q -O "$WORKFLOW_DIR/Wan_2_2_Animate_tracker_Head_Swap_v01.json" \
        "https://raw.githubusercontent.com/Herony82/ComfyUI-WanAnimatePreprocess-track/main/workflows/Wan_2_2_-_Animate__tracker__-_Head_Swap_-_v01.json"
fi

# ────────────────────────────────────────────────────────────
# 6. Avvia ComfyUI
# ────────────────────────────────────────────────────────────
log "Avvio ComfyUI su porta 8188..."
cd "$COMFYUI_DIR"
python main.py \
    --listen 0.0.0.0 \
    --port 8188 \
    --enable-cors-header \
    --preview-method auto
