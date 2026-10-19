"""
Rilevamento peronospora su foglia di vite - versione Streamlit.

  - Upload immagine (tif/png/jpg)
  - Due viste: originale | rilevamento (con perimetro della foglia)
  - Slider per i 3 parametri principali
  - Correzione manuale:
      * clic su una chiazza -> la esclude
      * poligono (clic sinistro = vertici, clic destro = chiudi) -> aggiunge
        una zona da considerare come fungo
  - Annulla / Reset (il Reset ripristina anche gli slider)
  - Risultato: copertura percentuale della foglia
  - Download overlay, maschera, CSV riassuntivo

Richiede: numpy scipy scikit-image opencv-python-headless pillow
          streamlit pandas
Cartella necessaria accanto a questo file: vite_viewer/index.html
"""
import io
import os
import hashlib
import base64
import numpy as np
import pandas as pd
import streamlit as st
import cv2
from PIL import Image
from scipy import ndimage as ndi
from skimage import segmentation
import streamlit.components.v1 as components

# ------------------------- rilevamento (invariato) -------------------------
FIXED = {'vein_r': 8, 'tight_W': 0.84, 'split': 15}

# valori predefiniti degli slider (usati anche dal pulsante Reset)
DEFAULTS = {'seed_W': 0.88, 'dens': 0.14, 'area_min': 200}

RED = np.array([255, 0, 0])
LEAF_EDGE = (144, 238, 144)      # verde chiaro


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


# ------------------------- correzioni e overlay ----------------------------
def apply_actions(seg_raw, actions, leaf):
    """Riapplica in ordine le correzioni manuali alla segmentazione grezza.

    actions: lista di dict
      {'t': 'rm',  'pos': (x, y)}          -> elimina la chiazza in (x, y)
      {'t': 'add', 'pts': [[fx, fy], ...]} -> aggiunge un poligono (coord. 0-1)
    """
    seg = seg_raw.copy()
    H, W = seg.shape
    nxt = int(seg.max()) + 1
    for a in actions:
        if a['t'] == 'rm':
            x, y = a['pos']
            lab = seg[y, x]
            if lab:
                seg[seg == lab] = 0
        else:
            pts = np.array([[round(px * (W - 1)), round(py * (H - 1))]
                            for px, py in a['pts']], np.int32)
            m = np.zeros((H, W), np.uint8)
            cv2.fillPoly(m, [pts], 1)
            m = (m > 0) & leaf
            if m.any():
                seg[m] = nxt
                nxt += 1
    return seg


def leaf_contours(mask):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                             cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in cs if len(c) > 20]


def make_overlay(base, seg, contours=None, thick=2):
    ov = np.ascontiguousarray(base.copy())
    m = seg > 0
    ov[m] = (0.55 * ov[m] + 0.45 * RED).astype(np.uint8)
    edge = segmentation.find_boundaries(seg, mode='outer')
    ov[edge] = RED
    if contours:
        cv2.drawContours(ov, contours, -1, LEAF_EDGE, thick, cv2.LINE_AA)
    return ov


def png_bytes(arr):
    b = io.BytesIO()
    Image.fromarray(arr).save(b, format='PNG', compress_level=1)
    return b.getvalue()


# ------------------------------- guida -------------------------------------
GUIDA_USO = """\
**1. Carica l'immagine** dalla barra laterale (TIF, PNG o JPG).

**2. Regola i parametri** finché le zone rosse coprono bene le macchie bianche.
Il tasto **Reset** riporta gli slider ai valori predefiniti.

**3. Correggi a mano** dalla barra strumenti sopra l'immagine:

- **Rimuovi chiazza**: clicca su una zona rossa per escluderla.
- **Aggiungi zona**: clicca per segnare i vertici di un poligono; con il
  **tasto destro** (o *Invio*) il poligono si chiude e l'area viene contata
  come fungo. *Esc* annulla il disegno, *Backspace* toglie l'ultimo vertice.
- **Naviga**: trascina per spostare l'immagine, rotellina per lo zoom. In ogni
  modalità puoi spostare l'immagine trascinando.

**4. Leggi la copertura** nelle schede in alto: è la percentuale di area della
foglia occupata dal fungo (la linea verde chiaro è il perimetro della foglia).

**5. Esporta** overlay, maschera e CSV dalla scheda *Esportazione*.
"""

GUIDA_ALGO = """\
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
Distance transform + massimi locali + watershed per separare chiazze unite
(serve solo a poter escludere le chiazze una per una).

RIEPILOGO SLIDER
1) Bianchezza: alto = meno falsi positivi
2) Densità:    alto = servono grumi fitti
3) Area:       alto = via le macchiette
"""

# --------------------------------- UI --------------------------------------
MAX_DISP = 2000   # lato massimo (px) delle immagini inviate al visualizzatore

st.set_page_config(page_title="Detect Vite - Peronospora",
                   layout="wide",
                   initial_sidebar_state="expanded")

_viewer = components.declare_component(
    "vite_viewer",
    path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "vite_viewer"))

STYLE = """
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2.5rem; max-width: 1600px; }

    .app-header { border-bottom: 1px solid rgba(128,128,128,.35);
                  padding-bottom: .8rem; margin-bottom: 1.1rem; }
    .app-header h1 { font-size: 1.6rem; font-weight: 650; letter-spacing: -.01em;
                     margin: 0; padding: 0; }
    .app-header p { margin: .25rem 0 0 0; font-size: .92rem; opacity: .7; }

    .section-title { font-size: .76rem; font-weight: 650; letter-spacing: .08em;
                     text-transform: uppercase; opacity: .65; margin: 0 0 .5rem 0; }

    /* sidebar */
    section[data-testid="stSidebar"] .block-container { padding-top: 1.4rem; }
    section[data-testid="stSidebar"] .section-title { margin-top: .4rem; }
    section[data-testid="stSidebar"] hr { margin: 1rem 0; }

    /* metriche */
    div[data-testid="stMetric"] { background: rgba(128,128,128,.07);
        border: 1px solid rgba(128,128,128,.25); border-radius: 8px;
        padding: .8rem 1.1rem; }
    div[data-testid="stMetricLabel"] p { font-size: .76rem; font-weight: 600;
        letter-spacing: .04em; text-transform: uppercase; opacity: .7; }
    div[data-testid="stMetricValue"] { font-size: 1.7rem; font-weight: 650; }

    button[data-baseweb="tab"] { font-weight: 600; padding-left: 1.1rem;
                                 padding-right: 1.1rem; }
    .stButton > button, .stDownloadButton > button { border-radius: 6px;
        font-weight: 550; width: 100%; }

    .legend { display: flex; gap: 1.4rem; flex-wrap: wrap; font-size: .85rem;
              opacity: .8; margin-top: .5rem; }
    .legend i { display: inline-block; width: 22px; height: 0; vertical-align: middle;
                margin-right: 6px; border-top: 3px solid; }

    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)


def section_title(text):
    st.markdown(f'<div class="section-title">{text}</div>',
                unsafe_allow_html=True)


# ------------------------- funzioni con cache ------------------------------
def _data_url(arr, quality=90):
    b = io.BytesIO()
    Image.fromarray(arr).save(b, format='JPEG', quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


@st.cache_resource(max_entries=2, show_spinner="Caricamento immagine...")
def load_bundle(img_key, _data):
    """Decodifica e preparazione: eseguite una sola volta per immagine."""
    arr = np.array(Image.open(io.BytesIO(_data)).convert('RGB'))
    leaf, inside, W, V = prepara(arr.astype(np.float32))
    H, W_img = arr.shape[:2]
    sc = min(1.0, MAX_DISP / max(H, W_img))
    if sc < 1.0:
        disp = cv2.resize(arr, (int(W_img * sc), int(H * sc)),
                          interpolation=cv2.INTER_AREA)
        leaf_d = cv2.resize(leaf.astype(np.uint8),
                            (disp.shape[1], disp.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
    else:
        disp, leaf_d = arr, leaf
    return {'arr': arr, 'leaf': leaf, 'inside': inside, 'W': W, 'V': V,
            'disp': disp, 'orig_url': _data_url(disp),
            'leaf_cnt_disp': leaf_contours(leaf_d)}


@st.cache_resource(max_entries=12, show_spinner="Analisi in corso...")
def cached_detect(img_key, seed_W, dens, area_min, _b):
    params = {'seed_W': seed_W, 'dens': dens, 'area_min': area_min, **FIXED}
    return detect(params, _b['leaf'], _b['inside'], _b['W'], _b['V'])


# ------------------------------- stato -------------------------------------
if "actions" not in st.session_state:
    st.session_state.actions = []
if "file_id" not in st.session_state:
    st.session_state.file_id = None
if "export" not in st.session_state:
    st.session_state.export = None


def reset_all():
    """Riporta slider ai valori predefiniti e cancella le correzioni."""
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    st.session_state.actions = []
    st.session_state.export = None


def undo_last():
    if st.session_state.actions:
        st.session_state.actions.pop()


# ------------------------------- sidebar -----------------------------------
with st.sidebar:
    section_title("1 · Immagine")
    file = st.file_uploader("Immagine foglia",
                            type=['tif', 'tiff', 'png', 'jpg', 'jpeg'],
                            label_visibility="collapsed")

    st.divider()
    section_title("2 · Parametri di rilevamento")
    seed_W = st.slider("Bianchezza minima", 0.80, 0.95, DEFAULTS['seed_W'],
                       0.005, key="seed_W",
                       help="Valori alti riducono i falsi positivi.")
    dens = st.slider("Densità minima di pixel bianchi vicini", 0.05, 0.30,
                     DEFAULTS['dens'], 0.01, key="dens",
                     help="Valori alti richiedono grumi più fitti.")
    area_min = st.slider("Area minima di una chiazza (px)", 100, 3000,
                         DEFAULTS['area_min'], 50, key="area_min",
                         help="Valori alti eliminano le macchie più piccole.")

    st.divider()
    section_title("3 · Correzione manuale")
    st.caption("Usa gli strumenti sopra l'immagine: *Rimuovi chiazza* "
               "per escludere, *Aggiungi zona* per disegnare un poligono "
               "(clic destro per chiuderlo).")
    corr_info = st.empty()
    b1, b2 = st.columns(2)
    b1.button("Annulla", on_click=undo_last,
              help="Annulla l'ultima correzione manuale.")
    b2.button("Reset", on_click=reset_all,
              help="Ripristina i parametri predefiniti e cancella tutte "
                   "le correzioni manuali.")

    st.divider()
    section_title("Visualizzazione")
    show_perim = st.checkbox("Mostra perimetro della foglia", value=True,
                             key="show_perim")

    with st.expander("Calibrazione (opzionale)"):
        px_mm = st.number_input("mm per pixel (0 = misure in px)",
                                min_value=0.0, value=0.0, step=0.01,
                                key="px_mm")

# ------------------------------- caricamento -------------------------------
if file is None:
    st.markdown(
        """
        _____________________________________________________________________________
        1. **Carica una foto** della foglia dalla barra laterale (TIF, PNG o JPG).
        2. **Regola i parametri** finché le zone rosse coprono le macchie bianche.
        3. **Correggi a mano** dove serve: escludi chiazze o disegna zone da
           considerare come fungo.
        4. **Leggi la copertura percentuale** ed esporta i risultati.
        """
    )
    st.stop()

data = file.getvalue()
img_key = hashlib.md5(data).hexdigest()

# nuova immagine -> azzera le correzioni
if st.session_state.file_id != img_key:
    st.session_state.file_id = img_key
    st.session_state.actions = []
    st.session_state.export = None

bundle = load_bundle(img_key, data)
img_arr = bundle['arr']
leaf = bundle['leaf']
H, W_img = img_arr.shape[:2]
disp_h, disp_w = bundle['disp'].shape[:2]

seg_raw = cached_detect(img_key, seed_W, dens, area_min, bundle)
seg = apply_actions(seg_raw, st.session_state.actions, leaf)

# ----------------------- gestione eventi dal visualizzatore ----------------
VIEW_KEY = f"viewer_{img_key}"
LAST_KEY = f"last_event_{img_key}"


def handle_event(val, seg):
    """Applica un evento del visualizzatore. True se lo stato e' cambiato."""
    if not val or val.get("id") == st.session_state.get(LAST_KEY):
        return False
    st.session_state[LAST_KEY] = val.get("id")

    if val.get("type") == "poly":
        pts = [[min(max(float(x), 0.0), 1.0), min(max(float(y), 0.0), 1.0)]
               for x, y in val.get("pts", [])]
        if len(pts) >= 3:
            st.session_state.actions.append({'t': 'add', 'pts': pts})
            return True
        return False

    # rimozione chiazza
    x = min(max(int(val["fx"] * W_img), 0), W_img - 1)
    y = min(max(int(val["fy"] * H), 0), H - 1)
    r = max(2, int(5 * W_img / disp_w))          # tolleranza di aggancio
    y0, y1 = max(0, y - r), min(H, y + r + 1)
    x0, x1 = max(0, x - r), min(W_img, x + r + 1)
    win = seg[y0:y1, x0:x1]
    yy, xx = np.nonzero(win)
    if len(yy) == 0:
        return False
    i = np.argmin((yy + y0 - y) ** 2 + (xx + x0 - x) ** 2)
    pos = (int(xx[i] + x0), int(yy[i] + y0))
    st.session_state.actions.append({'t': 'rm', 'pos': pos})
    return True


# il click e' gia' nello stato del widget all'inizio dell'esecuzione
if handle_event(st.session_state.get(VIEW_KEY), seg):
    seg = apply_actions(seg_raw, st.session_state.actions, leaf)

n_add = sum(a['t'] == 'add' for a in st.session_state.actions)
n_rm = sum(a['t'] == 'rm' for a in st.session_state.actions)
corr_info.caption(f"Correzioni: **{n_add}** zone aggiunte · "
                  f"**{n_rm}** chiazze escluse")

# ------------------------------- metriche ----------------------------------
k = (px_mm ** 2) if px_mm > 0 else 1
unit = 'mm²' if px_mm > 0 else 'px'
fmt = (lambda v: f"{v:,.1f}") if px_mm > 0 else (lambda v: f"{int(v):,}")

leaf_px = int(leaf.sum()) or 1
fung_px = int((seg > 0).sum())
cop = 100 * fung_px / leaf_px

st.caption(f"File: {file.name}   |   Dimensioni: {W_img} × {H} px")

c1, c2, c3 = st.columns(3)
c1.metric("Copertura della foglia", f"{cop:.2f} %")
c2.metric("Area interessata", f"{fmt(fung_px * k)} {unit}")
c3.metric("Area foglia", f"{fmt(leaf_px * k)} {unit}")

st.write("")

# overlay per la visualizzazione (a risoluzione ridotta: molto più veloce)
if (disp_w, disp_h) != (W_img, H):
    seg_d = cv2.resize(seg, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)
else:
    seg_d = seg
ov_d = make_overlay(bundle['disp'], seg_d,
                    bundle['leaf_cnt_disp'] if show_perim else None, thick=2)
overlay_id = hashlib.md5(repr((img_key, seed_W, dens, area_min, show_perim,
                               st.session_state.actions)).encode()
                         ).hexdigest()[:12]

# --------------------------------- tab -------------------------------------
tab_an, tab_exp, tab_guida = st.tabs(["Analisi", "Esportazione", "Guida"])

with tab_an:
    res = _viewer(original=bundle['orig_url'], overlay=_data_url(ov_d),
                  original_id=img_key, overlay_id=overlay_id,
                  height=740, key=VIEW_KEY, default=None)
    st.markdown(
        '<div class="legend">'
        '<span><i style="border-color:#ff4b4b"></i>Zone considerate fungo</span>'
        '<span><i style="border-color:#90ee90"></i>Perimetro della foglia</span>'
        '</div>', unsafe_allow_html=True)
    # sicurezza: se il click non era ancora nello stato, applicalo e ridisegna
    if handle_event(res, seg):
        st.rerun()

with tab_exp:
    section_title("File di output")
    sig = (img_key, seed_W, dens, area_min, px_mm, show_perim,
           repr(st.session_state.actions))
    if st.button("Genera / aggiorna file di esportazione"):
        thick = max(2, int(max(H, W_img) / 600))
        ov_full = make_overlay(img_arr, seg,
                               leaf_contours(leaf) if show_perim else None,
                               thick)
        summary = pd.DataFrame([{
            'file': file.name,
            f'area_foglia_{unit}': round(leaf_px * k, 3),
            f'area_interessata_{unit}': round(fung_px * k, 3),
            'copertura_percentuale': round(cop, 3),
            'bianchezza_min': seed_W,
            'densita_min': dens,
            'area_min_chiazza_px': area_min,
            'mm_per_pixel': px_mm,
            'zone_aggiunte': n_add,
            'chiazze_escluse': n_rm,
        }])
        st.session_state.export = {
            'sig': sig,
            'overlay': png_bytes(ov_full),
            'mask': png_bytes(((seg > 0) * 255).astype(np.uint8)),
            'csv': summary.to_csv(index=False),
        }
    ex = st.session_state.export
    if ex is None or ex['sig'] != sig:
        st.info("I file non sono aggiornati: premi il pulsante sopra per "
                "generarli con i parametri e le correzioni correnti.")
    else:
        d1, d2, d3 = st.columns(3, gap="medium")
        with d1:
            with st.container(border=True):
                st.markdown("**Overlay**")
                st.caption("Immagine con zone e perimetro evidenziati (PNG).")
                st.download_button("Scarica overlay", ex['overlay'],
                                   file_name="overlay.png", mime="image/png")
        with d2:
            with st.container(border=True):
                st.markdown("**Maschera**")
                st.caption("Maschera binaria delle zone con fungo (PNG).")
                st.download_button("Scarica maschera", ex['mask'],
                                   file_name="maschera.png", mime="image/png")
        with d3:
            with st.container(border=True):
                st.markdown("**Riepilogo**")
                st.caption("Copertura percentuale, aree e parametri (CSV).")
                st.download_button("Scarica CSV", ex['csv'],
                                   file_name="copertura.csv", mime="text/csv")

with tab_guida:
    section_title("Come si usa")
    with st.container(border=True):
        st.markdown(GUIDA_USO)
    with st.expander("Come funziona l'algoritmo di rilevamento"):
        st.text(GUIDA_ALGO)