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

GUIDA_PANORAMICA = """\
L'app non usa reti neurali: applica una catena di **operazioni classiche di
elaborazione d'immagine** che sfrutta tre fatti: lo sfondo è nero, la foglia è
verde e la peronospora forma **macchie biancastre** (R, G e B simili e alti).

```
immagine RGB
   │
   ├─ 0  Preparazione ─ maschera foglia, maschera interna, mappe W e V
   ├─ 1  Pixel candidati (soglia su W e V)             ◄ slider "Bianchezza"
   ├─ 2  Eliminazione nervature (forma + apertura)
   ├─ 3  Raggruppamento (densità locale)               ◄ slider "Densità"
   ├─ 4  Contorno preciso (soglia + chiusura)
   ├─ 5  Filtro area minima                            ◄ slider "Area minima"
   ├─ 6  Separazione chiazze unite (watershed)
   ├─ 7  Correzioni manuali (rimozione / poligoni)
   └─ 8  Copertura % = pixel fungo / pixel foglia × 100
```

Quasi tutte le fasi lavorano su **maschere binarie** (immagini con pixel 0/1:
"è foglia / non è foglia", "è fungo / non è fungo"). Le operazioni usate
vengono spiegate nel glossario in fondo.
"""

GUIDA_FASI = [
    ("Fase 0 · Preparazione: maschere e mappe di colore", r"""
Eseguita **una sola volta** per immagine, a risoluzione piena.

**Maschera "foglia"**
1. Si tiene ogni pixel in cui il canale più luminoso supera 25
   (`max(R,G,B) > 25`): tutto ciò che non è nero.
2. Si eliminano le **componenti connesse** (gruppi di pixel adiacenti, anche in
   diagonale) più piccole di 5000 px: polvere, rumore del sensore, riflessi.
3. Si **riempiono i buchi** interni: un'area scura dentro la foglia (per
   esempio un foro) resta comunque foglia.

È la maschera su cui si calcola il denominatore della copertura e da cui si
traccia la **linea verde chiaro** del perimetro.

**Maschera "interno"**
La maschera foglia viene **erosa** con un elemento 3×3 ripetuto 12 volte,
cioè ridotta di circa 12 px su tutto il bordo. Il margine della foglia è spesso
più chiaro (riflessi, bordo secco) e verrebbe scambiato per fungo: escluderlo
a monte evita molti falsi positivi.

**Mappa di bianchezza W**

$$W = \frac{\min(R,\,B)}{G + 1}$$

- su un pixel **bianco** R ≈ G ≈ B, quindi W ≈ 1;
- su un pixel **verde** G è molto maggiore di R e B, quindi W resta basso
  (tipicamente 0.3–0.6);
- usare il *minimo* tra R e B impedisce che pixel gialli o violetti risultino
  "bianchi";
- il +1 evita la divisione per zero.

La mappa viene poi sfocata con un **filtro gaussiano** (σ = 1 px) per
attenuare il rumore pixel per pixel.

**Mappa di luminosità V**
Media dei tre canali, anch'essa sfocata (σ = 1). Serve a distinguere il bianco
(chiaro) dal grigio o dall'ombra (scuro) quando W da solo non basta.
"""),
    ("Fase 1 · Pixel candidati  [slider: Bianchezza minima]", r"""
Una **sogliatura** (thresholding) a tre condizioni:

```
candidato = (W > soglia_bianchezza)  e  (V > 135)  e  dentro "interno"
```

Poi si scartano i gruppi di meno di 7 px (granelli isolati).

Il risultato sono i **granelli bianchi**: non ancora chiazze, solo i punti che
hanno il colore giusto.

- **Alzando** la soglia (0.88 → 0.92) restano solo i pixel molto bianchi:
  meno falsi positivi, ma si perdono le macchie più sbiadite.
- **Abbassandola** si includono anche zone verdastre o grigie.
"""),
    ("Fase 2 · Eliminazione delle nervature", r"""
Le nervature della foglia sono chiare e possono superare la soglia di W.
Si riconoscono dalla **forma**: sono lunghe e sottili.

1. I granelli vengono **etichettati** (ogni componente connessa riceve un
   numero).
2. Per ogni componente con area > 150 px si calcola la **matrice di
   covarianza** delle coordinate (x, y) dei suoi pixel e i suoi due autovalori
   λ₁ ≥ λ₂. Questi misurano l'allungamento della forma lungo i due assi.
3. Si ricava l'**eccentricità**:

$$e = \sqrt{1 - \frac{\lambda_2}{\lambda_1}}$$

   - cerchio → e ≈ 0
   - linea molto sottile → e ≈ 1
4. Le componenti con **e > 0.97** sono considerate nervature.
5. Su queste si applica una **apertura morfologica** con elemento ellittico di
   raggio 8 px: elimina le strutture più strette di circa 17 px e conserva i
   blob più larghi. Una nervatura sparisce; una chiazza che la attraversa
   rimane.

La maschera finale dei candidati è: *(candidati non allungati) ∪ (parti
spesse delle forme allungate)*, di nuovo ripulita dai gruppi < 7 px.
"""),
    ("Fase 3 · Raggruppamento per densità  [slider: Densità minima]", r"""
Una macchia vera è un **grumo fitto** di granelli bianchi; i falsi positivi
sono granelli sparsi. Per misurarlo:

1. Si calcola la **densità locale** con un filtro a media (*box filter*) su una
   finestra 41 × 41 px: in ogni punto, la **frazione** di pixel candidati
   nell'intorno (valore tra 0 e 1).
2. Si tiene dove la densità supera la soglia e si è dentro "interno".
3. **Apertura** (raggio 5) per tagliare filamenti e ponti sottili.
4. Si scartano le zone < 500 px e si **riempiono i buchi**.

Il risultato è una **zona candidata**: un'area generosa che contiene la
chiazza, ma con un bordo ancora approssimativo (la finestra 41 × 41 sfoca i
confini).

- **Alzando** la densità servono grumi più compatti.
- **Abbassandola** anche i granelli poco raggruppati formano zone.
"""),
    ("Fase 4 e 5 · Contorno preciso e area minima  [slider: Area minima]", r"""
Per far aderire il bordo al bianco reale si interseca la zona candidata con
una soglia **più severa**:

```
stretta = zona_candidata  e  (W > 0.84)  e  (V > 120)
```

Poi una **chiusura morfologica** (raggio 4) richiude i piccoli buchi dentro le
macchie, e infine si scartano le componenti con area inferiore alla soglia
scelta nello slider.

- **Alzando** l'area minima spariscono le macchioline.
- **Abbassandola** si tengono anche i punti piccoli.

Questa è la **maschera del fungo** prima delle correzioni manuali.
"""),
    ("Fase 6 · Separazione delle chiazze unite (watershed)", r"""
Chiazze vicine possono risultare fuse in un'unica componente. Per poterle
escludere una per una si separano così:

1. **Trasformata di distanza** (distance transform): ogni pixel della maschera
   prende il valore della distanza dal bordo più vicino. Il centro di una
   chiazza tonda è un "picco".
2. Si cercano i **massimi locali** della distanza in una finestra di 31 × 31
   px (e distanza ≥ 5): ogni massimo diventa un **seme**, uno per chiazza.
3. **Watershed** (spartiacque): l'immagine delle distanze è vista come un
   rilievo e "riempita d'acqua" a partire dai semi; dove le acque di due semi
   si incontrano si traccia il confine.
4. Fuori dalla maschera le etichette sono azzerate.

Questa fase **non cambia la copertura totale**: serve solo a dare un'identità
a ogni chiazza, così lo strumento *Rimuovi chiazza* esclude solo quella
cliccata.
"""),
    ("Fase 7 · Correzioni manuali", r"""
Le correzioni vengono **riapplicate in ordine** alla segmentazione grezza ogni
volta che qualcosa cambia (per questo *Annulla* funziona in modo affidabile).

- **Rimuovi chiazza**: il clic viene convertito nelle coordinate dell'immagine
  originale e si cerca il pixel di fungo più vicino entro una piccola
  tolleranza; tutta l'etichetta a cui appartiene viene azzerata.
- **Aggiungi zona**: i vertici sono salvati come coordinate **normalizzate**
  (0–1), quindi indipendenti dalla risoluzione di visualizzazione. Il
  poligono è **rasterizzato** a risoluzione piena (riempimento scanline),
  **intersecato con la maschera foglia** (non si può contare fungo fuori dalla
  foglia) e aggiunto come nuova etichetta, sovrascrivendo ciò che c'era sotto.
- Si può rimuovere anche una zona disegnata: i clic di rimozione sono
  applicati dopo le aggiunte precedenti.
"""),
    ("Fase 8 · Calcolo della copertura e visualizzazione", r"""
**Copertura**

$$\text{copertura}\,[\%] = 100 \times \frac{\#\text{pixel con fungo}}{\#\text{pixel della foglia}}$$

Il denominatore è l'**intera** maschera foglia (non quella erosa). Con una
calibrazione di *mm per pixel* le aree sono moltiplicate per (mm/px)².

**Overlay**
- le zone con fungo sono **miscelate in rosso** (55 % immagine + 45 % rosso)
  con un contorno rosso pieno, ricavato dai **bordi** tra etichette;
- il **perimetro della foglia** è il contorno esterno della maschera foglia
  (*find contours*, solo le curve esterne), disegnato in verde chiaro con
  **anti-aliasing**.

**Prestazioni**
Il rilevamento gira a risoluzione piena e viene messo in **cache**: cambiare
una correzione non lo ripete. Al visualizzatore invece viene inviata
un'immagine ridotta a un massimo di 2000 px per lato, per restare fluidi;
l'esportazione usa invece la risoluzione originale.
"""),
]

GUIDA_GLOSSARIO = r"""
| Termine | Che cosa fa |
|---|---|
| **Sogliatura** | Trasforma una mappa in una maschera 0/1: 1 dove il valore supera una soglia. |
| **Componente connessa** | Gruppo di pixel a 1 adiacenti (anche in diagonale). Si può contare, misurare o eliminare. |
| **Erosione** | Restringe le forme di un certo raggio; elimina i dettagli sottili. |
| **Dilatazione** | Allarga le forme di un certo raggio. |
| **Apertura** | Erosione + dilatazione: elimina rumore e strutture sottili senza ridurre le forme grandi. |
| **Chiusura** | Dilatazione + erosione: chiude piccoli buchi e fessure. |
| **Filtro gaussiano** | Sfocatura pesata; riduce il rumore. σ è l'ampiezza in pixel. |
| **Box filter** | Media su una finestra quadrata; qui misura la densità locale. |
| **Autovalori / eccentricità** | Descrivono quanto una forma è allungata. |
| **Trasformata di distanza** | Ogni pixel prende la distanza dal bordo più vicino. |
| **Watershed** | Segmentazione "a spartiacque": separa oggetti uniti a partire da semi. |
| **Rasterizzazione** | Conversione di un poligono in pixel pieni. |
"""

GUIDA_CONSIGLI = """\
| Problema | Cosa provare |
|---|---|
| Troppe zone rosse sul verde sano | Alza **Bianchezza** (es. 0.90) o **Densità** |
| Macchie evidenti non rilevate | Abbassa **Bianchezza** o **Densità** |
| Tante macchioline sparse | Alza **Area minima** |
| Macchie piccole ma vere escluse | Abbassa **Area minima** |
| Nervature segnate come fungo | Alza **Densità**; in alternativa escludile con *Rimuovi chiazza* |
| Fungo non rilevato in una zona precisa | Disegnala con *Aggiungi zona* |
| Il bordo della foglia risulta rosso | Normale se il bordo è molto chiaro: escludilo con *Rimuovi chiazza* |

**Condizioni di ripresa consigliate:** sfondo nero uniforme, foglia ben
distesa, luce diffusa senza riflessi speculari, stessa distanza e stessa
illuminazione tra le foto che vuoi confrontare (le soglie dipendono dal
colore).
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
        #### ____________________________________________________________________
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

    st.write("")
    section_title("Processo di elaborazione dell'immagine")
    with st.container(border=True):
        st.markdown(GUIDA_PANORAMICA)
    for titolo, testo in GUIDA_FASI:
        with st.expander(titolo):
            st.markdown(testo)

    st.write("")
    section_title("Consigli per la messa a punto")
    with st.container(border=True):
        st.markdown(GUIDA_CONSIGLI)

    st.write("")
    section_title("Glossario delle operazioni")
    with st.container(border=True):
        st.markdown(GUIDA_GLOSSARIO)