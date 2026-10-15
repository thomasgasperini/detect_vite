"""
Rilevamento peronospora su foglia di vite - versione Streamlit.

  - Upload immagine (tif/png/jpg)
  - Due viste: originale | overlay con chiazze
  - Slider per i 3 parametri principali
  - Click su overlay per rimuovere una chiazza (streamlit-image-coordinates)
  - Annulla / Reset
  - Download overlay, maschera, CSV
  - Guida integrata

Richiede: numpy scipy scikit-image opencv-python-headless pillow
          streamlit streamlit-image-coordinates pandas
"""
import io
import csv
import numpy as np
import pandas as pd
import streamlit as st
import cv2
from PIL import Image
from scipy import ndimage as ndi
from skimage import measure, segmentation
from streamlit_image_coordinates import streamlit_image_coordinates

# ------------------------- rilevamento (invariato) -------------------------
FIXED = {'vein_r': 8, 'tight_W': 0.84, 'split': 15}

def rm_small(mask, n):
    lab, nlab = ndi.label(mask, structure=np.ones((3, 3), int))
    if nlab == 0:
        return mask.astype(bool)
    counts = np.bincount(lab.ravel(), minlength=nlab + 1)
    keep = counts >= n
    keep[0] = False
    return keep[lab]

def morph(mask, r, op):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.morphologyEx(mask.astype(np.uint8), op, k) > 0

def prepara(a):
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    leaf = ndi.binary_fill_holes(rm_small(a.max(2) > 25, 5000))
    inside = cv2.erode(leaf.astype(np.uint8), np.ones((3, 3), np.uint8),
                       iterations=12) > 0
    W = cv2.GaussianBlur((np.minimum(R, B) / (G + 1)).astype(np.float32),
                         (0, 0), 1)
    V = cv2.GaussianBlur(a.mean(2).astype(np.float32), (0, 0), 1)
    return leaf, inside, W, V

def detect(p, leaf, inside, W, V):
    seed = rm_small((W > p['seed_W']) & (V > 135) & inside, 7)

    lab, nlab = ndi.label(seed, structure=np.ones((3, 3), int))
    if nlab:
        objs = ndi.find_objects(lab)
        ids = []
        for i, sl in enumerate(objs, 1):
            if sl is None:
                continue
            m = (lab[sl] == i)
            area = int(m.sum())
            if area <= 150:
                continue
            ys, xs = np.nonzero(m)
            if len(ys) < 3:
                continue
            cov = np.cov(np.vstack([ys, xs]))
            ev = np.linalg.eigvalsh(cov)
            if ev[1] <= 0:
                continue
            ecc = float(np.sqrt(1 - ev[0] / ev[1]))
            if ecc > 0.97:
                ids.append(i)
        elong = np.isin(lab, ids)
    else:
        elong = np.zeros_like(seed)

    seed = rm_small((seed & ~elong) |
                    morph(elong, int(p['vein_r']), cv2.MORPH_OPEN), 7)

    dens = cv2.boxFilter(seed.astype(np.float32), -1, (41, 41),
                         normalize=True)
    cand = ndi.binary_fill_holes(
        rm_small(morph((dens > p['dens']) & inside, 5, cv2.MORPH_OPEN), 500))
    tight = rm_small(morph(cand & (W > p['tight_W']) & (V > 120),
                           4, cv2.MORPH_CLOSE), int(p['area_min']))
    if not tight.any():
        return np.zeros_like(tight, dtype=np.int32)

    dist = cv2.distanceTransform(tight.astype(np.uint8), cv2.DIST_L2, 5)
    k = int(p['split'])
    win = 2 * k + 1
    mx = cv2.dilate(dist, cv2.getStructuringElement(cv2.MORPH_RECT,
                                                    (win, win)))
    peaks = (dist >= mx - 1e-6) & (dist >= 5) & tight
    ys, xs = np.nonzero(peaks)
    mk = np.zeros(tight.shape, np.int32)
    for i, (y, x) in enumerate(zip(ys, xs), 1):
        if mk[y, x] == 0:
            mk[y, x] = i
    dist_u8 = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    labels = cv2.watershed(cv2.cvtColor(dist_u8, cv2.COLOR_GRAY2BGR),
                           mk.astype(np.int32))
    labels[labels < 0] = 0
    labels[~tight] = 0
    return labels.astype(np.int32)

# ------------------------------- guida -------------------------------------
GUIDA = """\
COME FUNZIONA IL RILEVAMENTO DELLA PERONOSPORA
==============================================

IN DUE PAROLE
La peronospora forma MACCHIE BIANCHE sulla foglia. L'algoritmo sfrutta:
  - foglia VERDE
  - chiazze BIANCHE
  - sfondo NERO

FASE 0 - PREPARAZIONE
- maschera "foglia": pixel non neri, riempimento buchi, rimozione rumore
- maschera "interno": foglia erosa di 12 px (toglie il bordo chiaro)
- mappa W = min(R,B)/(G+1)  ->  ~1 se bianco, ~0 se verde
- mappa V = luminosità media, sfocata

FASE 1 - GRANULI CANDIDATI  [SLIDER 1]
Pixel con W > soglia, V > 135, dentro "interno". Scarta gruppi < 7 px.

FASE 2 - NERVATURE (fissa)
Forme lunghe e sottili (eccentricità > 0.97) rimosse con apertura r=8.

FASE 3 - RAGGRUPPAMENTO  [SLIDER 2]
Densità locale in finestra 41x41. Sopra soglia -> candidato chiazza.

FASE 4 - CONTORNO (fissa)
W > 0.84 + chiusura r=4 per aderire al bianco vero.

FASE 5 - AREA MINIMA  [SLIDER 3]
Scarta componenti sotto la soglia.

FASE 6 - WATERSHED (fissa)
Distance transform + massimi locali + watershed per separare chiazze unite.

RIEPILOGO SLIDER
1) Bianchezza: alto = meno falsi positivi
2) Densità:    alto = servono grumi fitti
3) Area:       alto = via le macchiette
"""

# --------------------------------- UI --------------------------------------
st.set_page_config(page_title="Detect Vite - Peronospora",
                   layout="wide",
                   initial_sidebar_state="expanded")

STYLE = """
<style>
    /* Contenitore principale */
    .block-container {
        padding-top: 2.2rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    /* Intestazione */
    .app-header {
        border-bottom: 1px solid rgba(128, 128, 128, 0.35);
        padding-bottom: 0.9rem;
        margin-bottom: 1.4rem;
    }
    .app-header h1 {
        font-size: 1.65rem;
        font-weight: 650;
        letter-spacing: -0.01em;
        margin: 0;
        padding: 0;
    }
    .app-header p {
        margin: 0.25rem 0 0 0;
        font-size: 0.92rem;
        opacity: 0.7;
    }

    /* Titoli di sezione */
    .section-title {
        font-size: 0.78rem;
        font-weight: 650;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        opacity: 0.65;
        margin: 0 0 0.6rem 0;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.6rem;
    }
    section[data-testid="stSidebar"] h2 {
        font-size: 1.05rem;
        font-weight: 650;
    }

    /* Metriche */
    div[data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.07);
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 6px;
        padding: 0.9rem 1.1rem;
    }
    div[data-testid="stMetricLabel"] p {
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        opacity: 0.7;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.7rem;
        font-weight: 650;
    }

    /* Tab */
    button[data-baseweb="tab"] {
        font-weight: 600;
        padding-left: 1.1rem;
        padding-right: 1.1rem;
    }

    /* Pulsanti */
    .stButton > button, .stDownloadButton > button {
        border-radius: 5px;
        font-weight: 550;
        width: 100%;
    }

    /* Immagini */
    div[data-testid="stImage"] img {
        border-radius: 4px;
        border: 1px solid rgba(128, 128, 128, 0.25);
    }

    /* Elementi di default da nascondere */
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)


def section_title(text):
    st.markdown(f'<div class="section-title">{text}</div>',
                unsafe_allow_html=True)


# stato
if "removed" not in st.session_state:
    st.session_state.removed = []
if "file_id" not in st.session_state:
    st.session_state.file_id = None

# intestazione
st.markdown(
    """
    <div class="app-header">
        <h1>Detect Vite</h1>
        <p>Rilevamento e quantificazione della peronospora su foglia di vite</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------- sidebar -----------------------------------
with st.sidebar:
    st.header("Parametri di rilevamento")
    st.caption("Le modifiche vengono applicate immediatamente all'analisi.")

    seed_W = st.slider("Bianchezza minima del pixel",
                       0.80, 0.95, 0.88, 0.005,
                       help="Valori alti riducono i falsi positivi.")
    dens = st.slider("Densità minima di pixel bianchi vicini",
                     0.05, 0.30, 0.14, 0.01,
                     help="Valori alti richiedono grumi più fitti.")
    area_min = st.slider("Area minima di una chiazza (px)",
                         100, 3000, 200, 50,
                         help="Valori alti eliminano le macchie più piccole.")

    st.divider()
    st.subheader("Calibrazione")
    px_mm = st.number_input("mm per pixel (0 = misure in px)",
                            min_value=0.0, value=0.0, step=0.01)

    st.divider()
    st.subheader("Correzione manuale")
    st.caption("Clicca su una chiazza nella vista di destra per escluderla.")
    b1, b2 = st.columns(2)
    if b1.button("Annulla"):
        if st.session_state.get("removed"):
            st.session_state.removed.pop()
    if b2.button("Reset"):
        st.session_state.removed = []

# ------------------------------- caricamento -------------------------------
section_title("Immagine di input")
file = st.file_uploader("Immagine foglia",
                        type=['tif', 'tiff', 'png', 'jpg', 'jpeg'],
                        label_visibility="collapsed")

if file is None:
    st.info("Carica un'immagine (TIF, PNG o JPG) per avviare l'analisi.")
    st.stop()

# reset rimozioni se cambia file
if st.session_state.file_id != file.name:
    st.session_state.file_id = file.name
    st.session_state.removed = []

img = Image.open(file).convert('RGB')
a = np.array(img).astype(np.float32)

# cache preparazione (dipende solo dall'immagine)
@st.cache_data(show_spinner=False)
def cached_prepara(img_bytes):
    arr = np.array(Image.open(io.BytesIO(img_bytes)).convert('RGB')).astype(np.float32)
    return prepara(arr)

buf = io.BytesIO()
img.save(buf, format='PNG')
leaf, inside, W, V = cached_prepara(buf.getvalue())

params = {'seed_W': seed_W, 'dens': dens, 'area_min': area_min, **FIXED}
seg_raw = detect(params, leaf, inside, W, V)

# applica rimozioni
seg = seg_raw.copy()
for x, y in st.session_state.removed:
    if 0 <= y < seg.shape[0] and 0 <= x < seg.shape[1]:
        lab = seg[y, x]
        if lab:
            seg[seg == lab] = 0

# overlay
ov = np.array(img).copy()
edge = segmentation.find_boundaries(seg, mode='outer')
ov[seg > 0] = (0.55 * ov[seg > 0] + 0.45 * np.array([255, 0, 0])).astype(np.uint8)
ov[edge] = (255, 0, 0)

# metriche
leaf_px = leaf.sum() or 1
n = int(seg.max())
cop = 100 * (seg > 0).sum() / leaf_px

H, W_img = ov.shape[:2]
st.caption(f"File: {file.name}   |   Dimensioni: {W_img} x {H} px")

section_title("Sintesi dell'analisi")
c1, c2, c3 = st.columns(3)
c1.metric("Chiazze rilevate", n)
c2.metric("Copertura foglia", f"{cop:.2f} %")
c3.metric("Rimozioni manuali", len(st.session_state.removed))

st.write("")

# CSV
k = (px_mm ** 2) if px_mm > 0 else 1
unit = 'mm2' if px_mm > 0 else 'px'
rows = []
for i, r in enumerate(measure.regionprops(seg), 1):
    rows.append({
        'id': i,
        f'area_{unit}': round(r.area * k, 3),
        'x_centroide_px': round(r.centroid[1]),
        'y_centroide_px': round(r.centroid[0]),
        'bbox_riga_min': r.bbox[0], 'bbox_col_min': r.bbox[1],
        'bbox_riga_max': r.bbox[2], 'bbox_col_max': r.bbox[3],
    })
df = pd.DataFrame(rows)

def png_bytes(arr):
    b = io.BytesIO()
    Image.fromarray(arr).save(b, format='PNG')
    return b.getvalue()

# --------------------------------- tab -------------------------------------
tab_an, tab_dati, tab_exp, tab_guida = st.tabs(
    ["Analisi", "Dettaglio chiazze", "Esportazione", "Guida"])

with tab_an:
    col1, col2 = st.columns(2, gap="large")
    with col1:
        section_title("Immagine originale")
        st.image(img, use_container_width=True)
    with col2:
        section_title("Chiazze rilevate - clicca per rimuovere")

        # ridimensiona per il widget click (max 900 px lato)
        scale = min(1.0, 900 / max(H, W_img))
        disp = cv2.resize(ov, (int(W_img * scale), int(H * scale)),
                          interpolation=cv2.INTER_AREA)
        coords = streamlit_image_coordinates(disp, key=f"click_{st.session_state.file_id}_{len(st.session_state.removed)}")
        if coords is not None:
            x = int(coords["x"] / scale)
            y = int(coords["y"] / scale)
            # raggio di tolleranza per agganciare la chiazza più vicina
            r = max(2, int(5 / scale))
            y0, y1 = max(0, y - r), min(H, y + r + 1)
            x0, x1 = max(0, x - r), min(W_img, x + r + 1)
            lab = seg[y, x] or seg[y0:y1, x0:x1].max()
            if lab:
                yy, xx = np.nonzero(seg == lab)
                i = np.argmin((xx - x) ** 2 + (yy - y) ** 2)
                pos = (int(xx[i]), int(yy[i]))
                if pos not in st.session_state.removed:
                    st.session_state.removed.append(pos)
                    st.rerun()

with tab_dati:
    section_title("Tabella delle chiazze")
    if df.empty:
        st.info("Nessuna chiazza rilevata con i parametri correnti.")
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)

with tab_exp:
    section_title("File di output")
    d1, d2, d3 = st.columns(3, gap="medium")
    with d1:
        with st.container(border=True):
            st.markdown("**Overlay**")
            st.caption("Immagine con le chiazze evidenziate (PNG).")
            st.download_button("Scarica overlay", png_bytes(ov),
                               file_name="overlay.png", mime="image/png")
    with d2:
        with st.container(border=True):
            st.markdown("**Maschera**")
            st.caption("Maschera binaria delle chiazze (PNG).")
            st.download_button("Scarica maschera",
                               png_bytes(((seg > 0) * 255).astype(np.uint8)),
                               file_name="maschera.png", mime="image/png")
    with d3:
        with st.container(border=True):
            st.markdown("**Dati**")
            st.caption("Area, centroide e bounding box di ogni chiazza (CSV).")
            csv_str = df.to_csv(index=False)
            st.download_button("Scarica CSV", csv_str,
                               file_name="chiazze.csv", mime="text/csv")

with tab_guida:
    section_title("Guida al rilevamento")
    with st.container(border=True):
        st.text(GUIDA)