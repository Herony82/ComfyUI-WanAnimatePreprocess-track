import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// ─────────────────────────────────────────────────────────────────────────────
//  Costanti UI
// ─────────────────────────────────────────────────────────────────────────────

const NODE_NAME   = "TrackedPoseAndFaceDetection";
const POINT_R     = 7;          // raggio cerchio punto (px)
const HIT_R       = 14;         // raggio di hit-test per rimozione (px)
const COLORS = [
    "#00ff88", "#ff4455", "#44aaff", "#ffcc00",
    "#ff44ff", "#44ffff", "#ff8800", "#aaffaa",
];

// ─────────────────────────────────────────────────────────────────────────────
//  Registrazione estensione
// ─────────────────────────────────────────────────────────────────────────────

app.registerExtension({
    name: "WanAnimatePreprocess.TrackPointPicker",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        // ── onNodeCreated ────────────────────────────────────────────────────
        const _onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            _onNodeCreated?.apply(this, arguments);
            _setupWidget(this);
        };

        // ── onExecuted: riceve il primo frame dal backend ────────────────────
        const _onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            _onExecuted?.apply(this, arguments);

            const info = message?.first_frame?.[0];
            if (info && this._pickerLoadFrame) {
                const url = api.apiURL(
                    `/view?filename=${encodeURIComponent(info.filename)}` +
                    `&subfolder=${encodeURIComponent(info.subfolder ?? "")}` +
                    `&type=${info.type ?? "temp"}&_t=${Date.now()}`
                );
                this._pickerLoadFrame(url);
            }
        };

        // ── onSerialize: aggiunge i punti al JSON del workflow ───────────────
        const _onSerialize = nodeType.prototype.onSerialize ?? function (o) { return o; };
        nodeType.prototype.onSerialize = function (o) {
            _onSerialize.call(this, o);
            // I punti sono già nel widget tracking_points — non serve altro
        };
    },
});

// ─────────────────────────────────────────────────────────────────────────────
//  Setup widget canvas nel nodo
// ─────────────────────────────────────────────────────────────────────────────

function _setupWidget(node) {
    // Trova il widget tracking_points e rendilo invisibile
    // (i valori vengono comunque serializzati nel workflow JSON)
    const tpWidget = node.widgets?.find(w => w.name === "tracking_points");
    if (tpWidget) {
        tpWidget.type            = "hidden";
        tpWidget.computeSize     = () => [0, -4];
    }

    // ── Costruisci DOM ───────────────────────────────────────────────────────
    const root = document.createElement("div");
    root.style.cssText = "width:100%; padding:0 4px 4px 4px; box-sizing:border-box;";

    // Istruzione
    const hint = document.createElement("div");
    hint.style.cssText = (
        "font-size:11px; color:#888; margin-bottom:5px; text-align:center; " +
        "line-height:1.5;"
    );
    hint.innerHTML = (
        "<b style='color:#aaa'>Shift+Click</b> aggiunge punto &nbsp;|&nbsp; " +
        "<b style='color:#aaa'>Click destro</b> sul punto lo rimuove<br>" +
        "<span style='color:#666'>Esegui una volta per caricare il frame</span>"
    );

    // Canvas
    const canvas = document.createElement("canvas");
    canvas.width  = 360;
    canvas.height = 202;   // placeholder 16:9
    canvas.style.cssText = (
        "width:100%; display:block; border-radius:5px; " +
        "border:1px solid #444; background:#111; cursor:crosshair;"
    );

    // Bottoni
    const btnRow = document.createElement("div");
    btnRow.style.cssText = "display:flex; gap:6px; margin-top:6px;";

    const undoBtn  = _makeBtn("↩ Undo",  "#333355");
    const clearBtn = _makeBtn("✕ Clear", "#552222");

    btnRow.append(undoBtn, clearBtn);
    root.append(hint, canvas, btnRow);

    node.addDOMWidget("_point_picker", "div", root, {
        getValue:  () => null,
        setValue:  ()  => {},
        serialize: false,
    });

    // ── Stato interno ────────────────────────────────────────────────────────
    // I punti sono memorizzati come coordinate NORMALIZZATE [0-1]
    // rispetto alle dimensioni dell'immagine originale.
    const state = {
        pts:    [],       // [{nx, ny}]  normalised
        imgW:   0,
        imgH:   0,
        bg:     null,     // HTMLImageElement
    };

    // Ripristina punti da workflow salvato
    if (tpWidget?.value && tpWidget.value !== "[]") {
        try {
            const saved = JSON.parse(tpWidget.value);
            if (Array.isArray(saved) && saved.length) {
                // Vengono convertiti in normalised dopo il caricamento dell'immagine
                state._pending = saved;   // [[px,py], ...]
            }
        } catch (_) { /* ignora */ }
    }

    // ── Rendering ────────────────────────────────────────────────────────────
    const ctx = canvas.getContext("2d");

    function draw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Background
        if (state.bg) {
            ctx.drawImage(state.bg, 0, 0, canvas.width, canvas.height);
        } else {
            ctx.fillStyle = "#111";
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.fillStyle    = "#555";
            ctx.font         = "13px sans-serif";
            ctx.textAlign    = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(
                "Esegui una volta per caricare il frame",
                canvas.width / 2, canvas.height / 2
            );
            return;
        }

        // Punti
        state.pts.forEach(({ nx, ny }, i) => {
            const px = nx * canvas.width;
            const py = ny * canvas.height;
            const col = COLORS[i % COLORS.length];

            // Ombra esterna
            ctx.shadowColor = "rgba(0,0,0,0.9)";
            ctx.shadowBlur  = 6;

            // Cerchio
            ctx.beginPath();
            ctx.arc(px, py, POINT_R, 0, Math.PI * 2);
            ctx.fillStyle   = col + "99";
            ctx.fill();
            ctx.strokeStyle = col;
            ctx.lineWidth   = 2;
            ctx.stroke();
            ctx.shadowBlur = 0;

            // Numero
            ctx.fillStyle    = "#fff";
            ctx.font         = `bold ${POINT_R + 3}px sans-serif`;
            ctx.textAlign    = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(i + 1, px, py);
        });
    }

    draw();   // mostra placeholder iniziale

    // ── Serializzazione → tpWidget ───────────────────────────────────────────
    function save() {
        if (!tpWidget) return;
        if (state.imgW === 0) { tpWidget.value = "[]"; return; }
        const px = state.pts.map(({ nx, ny }) => [
            Math.round(nx * state.imgW),
            Math.round(ny * state.imgH),
        ]);
        tpWidget.value = JSON.stringify(px);
    }

    // ── Hit-test: trova il punto più vicino al click (canvas coords) ─────────
    function nearestPoint(cx, cy) {
        let bestIdx = -1, bestD = Infinity;
        state.pts.forEach(({ nx, ny }, i) => {
            const px = nx * canvas.width;
            const py = ny * canvas.height;
            const d  = Math.hypot(px - cx, py - cy);
            if (d < bestD) { bestD = d; bestIdx = i; }
        });
        return bestD <= HIT_R ? bestIdx : -1;
    }

    // ── Mouse: Shift+click sinistro aggiunge, click destro rimuove ───────────
    canvas.addEventListener("click", e => {
        if (!state.bg) return;
        if (!e.shiftKey) return;   // solo Shift+click

        const r  = canvas.getBoundingClientRect();
        const cx = (e.clientX - r.left) * (canvas.width  / r.width);
        const cy = (e.clientY - r.top)  * (canvas.height / r.height);

        state.pts.push({
            nx: cx / canvas.width,
            ny: cy / canvas.height,
        });
        save();
        draw();
        app.graph.setDirtyCanvas(true, false);
    });

    canvas.addEventListener("contextmenu", e => {
        e.preventDefault();
        if (!state.bg || state.pts.length === 0) return;

        const r   = canvas.getBoundingClientRect();
        const cx  = (e.clientX - r.left) * (canvas.width  / r.width);
        const cy  = (e.clientY - r.top)  * (canvas.height / r.height);
        const idx = nearestPoint(cx, cy);

        if (idx >= 0) {
            state.pts.splice(idx, 1);
            save();
            draw();
            app.graph.setDirtyCanvas(true, false);
        }
    });

    // ── Undo / Clear ─────────────────────────────────────────────────────────
    undoBtn.addEventListener("click", () => {
        if (state.pts.length === 0) return;
        state.pts.pop();
        save();
        draw();
    });

    clearBtn.addEventListener("click", () => {
        if (state.pts.length === 0) return;
        state.pts = [];
        save();
        draw();
    });

    // ── Loader immagine (chiamato da onExecuted) ──────────────────────────────
    node._pickerLoadFrame = (url) => {
        const img = new Image();

        img.onload = () => {
            state.bg   = img;
            state.imgW = img.naturalWidth;
            state.imgH = img.naturalHeight;

            // Adatta altezza canvas all'aspect ratio reale
            const aspect  = img.naturalHeight / img.naturalWidth;
            canvas.height = Math.round(canvas.width * aspect);

            hint.innerHTML = (
                `<span style='color:#666'>${img.naturalWidth}×${img.naturalHeight}</span>` +
                " &nbsp;|&nbsp; " +
                "<b style='color:#aaa'>Shift+Click</b> aggiunge &nbsp;|&nbsp; " +
                "<b style='color:#aaa'>Click ⌋</b> rimuove" +
                `<br><span style='color:#00ff88'>${state.pts.length} punti</span>`
            );

            // Ripristina punti da pixel → normalised (workflow reload)
            if (state._pending) {
                state.pts = state._pending.map(([px, py]) => ({
                    nx: px / state.imgW,
                    ny: py / state.imgH,
                }));
                delete state._pending;
                // NON richiamare save(): i valori sono già nel tpWidget
            }

            draw();
            app.graph.setDirtyCanvas(true, false);
        };

        img.onerror = () => {
            console.warn("[TrackedPose] Impossibile caricare il primo frame:", url);
        };

        img.src = url;
    };

    // Aggiorna contatore punti ogni volta che si disegna
    const _origDraw = draw;
    // (l'hint viene aggiornato nel loader, non serve override qui)
}

// ─────────────────────────────────────────────────────────────────────────────
//  Utility
// ─────────────────────────────────────────────────────────────────────────────

function _makeBtn(label, bg) {
    const b = document.createElement("button");
    b.textContent = label;
    b.style.cssText = (
        `flex:1; padding:5px 0; background:${bg}; color:#ccc; ` +
        "border:1px solid #555; border-radius:4px; cursor:pointer; " +
        "font-size:12px; transition:opacity .15s;"
    );
    b.addEventListener("mouseenter", () => (b.style.opacity = "0.75"));
    b.addEventListener("mouseleave", () => (b.style.opacity = "1"));
    return b;
}
