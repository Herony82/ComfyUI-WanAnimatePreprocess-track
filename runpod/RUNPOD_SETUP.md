# RunPod Setup — ComfyUI Wan2.2 Animate Head Swap

## Requisiti GPU

| GPU | VRAM | Note |
|---|---|---|
| **A100 80GB** | 80 GB | Consigliato, gira senza offloading |
| **RTX 4090** | 24 GB | Funziona con block swap settings attuali |
| **A40** | 48 GB | Buon compromesso prezzo/performance |

Il modello Wan2.2 14B in BF16 occupa ~28 GB solo di pesi.
Con block swap (48 blocchi, come nel workflow) funziona su 24 GB VRAM con CPU offloading.
Serve però che il pod abbia **almeno 64 GB di RAM di sistema**.

---

## Step 1 — Network Volume (una volta sola)

1. RunPod → **Storage** → **+ Network Volume**
2. Nome: `wan22-models`
3. Size: **150 GB** (modelli ~45 GB + spazio output)
4. Region: stessa che userai per i pod
5. Crea il volume

Il Network Volume è persistente e costa ~$0.07/GB/mese.
I modelli (~45 GB) si scaricano **una volta sola** al primo avvio.

---

## Step 2 — Crea il Pod

1. RunPod → **Pods** → **+ GPU Pod**
2. **Template**: `RunPod Pytorch 2.2` (o simile base CUDA)
   - oppure cerca `ComfyUI` nei template della community
3. **GPU**: A100 80GB PCIe / RTX 4090 / A40
4. **Container Disk**: 20 GB (solo per OS e pip packages)
5. **Volume Disk**: collega il Network Volume creato sopra → mount su `/workspace`
6. **Expose ports**: aggiungi porta `8188` (TCP)
7. **On Start Command**:
   ```
   bash /workspace/start.sh
   ```
   (dopo il primo avvio, lo script è già nel volume)

---

## Step 3 — Primo avvio

Al primo avvio lo script:
- Clona ComfyUI
- Installa tutti i custom nodes
- Scarica tutti i modelli sul Network Volume (~45 GB, ci vuole tempo)
- Avvia ComfyUI sulla porta 8188

Puoi monitorare il progresso dal log del pod.

---

## Step 4 — Accedi all'UI

1. Nel pod → **Connect** → **HTTP Service [Port 8188]**
2. Si apre ComfyUI nel browser
3. Carica il workflow: `Wan_2_2_Animate_tracker_Head_Swap_v01.json`
   (è già nella cartella workflows di ComfyUI)

---

## Step 5 — Avvii successivi

Dalla seconda volta in poi:
- I modelli sono già sul volume (non si riscaricano)
- I custom nodes vengono solo aggiornati (`git pull`)
- Il pod è pronto in ~2-3 minuti

---

## Nota sul LoRA LightX2V

Il file scaricato si chiama:
`lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors`

Il workflow usa:
`wan21-lightx2v-i2v-14b-480p-cfg-step-distill-rank256-bf16.safetensors`

→ Aggiorna il nodo **WanVideoLoraSelectMulti** nel workflow con il nome file corretto dopo il primo download.

---

## Nota su GetNode / SetNode

Il workflow usa GetNode/SetNode per gestire variabili globali.
Identifica il pacchetto da cui provengono e aggiungilo a `start.sh`:
```bash
install_node "NOME_PACCHETTO" "https://github.com/..."
```

---

## Costi stimati (A100 80GB)

| Voce | Costo |
|---|---|
| GPU A100 80GB | ~$2.50/ora |
| Network Volume 150GB | ~$10.50/mese |
| Primo download modelli | ~1 ora di GPU |
| Generazione video (21 frame) | ~5-10 min |
