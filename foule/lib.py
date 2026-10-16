"""Bibliotheque de sequences detourees (EXR RGBA) : scan d'un dossier, cache rapide, vignettes."""
import csv
import hashlib
import os
import threading

import cv2
import numpy as np
import OpenEXR


# Mode anonymise (FOULE_ANON=1) : les personnages sont remplaces par des silhouettes unies aux couleurs de leur equipe,
# pour faire des captures / GIF sans aucune personne reconnaissable.
ANON = os.environ.get('FOULE_ANON') == '1'
TEAM_COL = ((70, 100, 175), (215, 120, 55))

MAX_H = 540     # hauteur maximale des personnages en memoire (assez pour un export 4K)


class Seq:
    def __init__(self, root, dirpath, files, fps, team=0, prefix=''):
        self.root = root
        self.path = dirpath
        self.team = team
        self.virtual = None          # definition d'une boucle creee par l'utilisateur
        rel = os.path.relpath(dirpath, root).replace('\\', '/')
        self.id = prefix + rel
        parts = rel.split('/')
        self.clip = parts[0] if len(parts) > 1 else '(racine)'
        self.name = parts[-1]
        self.files = files
        self.n = len(files)
        self.fps = fps
        self.frames = None          # (n, h, w, 4) uint8, alpha pre-multiplie
        self.lock = threading.Lock()
        self.thumb = None
        self.w = self.h = 0          # taille du recadrage d'origine (index.csv)

    @property
    def size(self):
        return None if self.frames is None else (self.frames.shape[2], self.frames.shape[1])


def _read_exr(path):
    return np.array(OpenEXR.File(path).channels()['RGBA'].pixels, dtype=np.float32)


def scan(root, team=0, prefix=''):
    """Trouve tous les dossiers contenant des .exr. fps lu dans index.csv si present."""
    fps_map = {}
    idx = os.path.join(root, 'index.csv')
    if os.path.exists(idx):
        with open(idx, encoding='utf-8') as f:
            for r in csv.DictReader(f):
                try:
                    fps_map[r['dossier'].replace('\\', '/')] = (float(r['fps']), int(r['largeur']), int(r['hauteur']))
                except (KeyError, ValueError):
                    pass
    seqs = []
    for dp, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        exr = sorted(f for f in files if f.lower().endswith('.exr'))
        if exr:
            rel = os.path.relpath(dp, root).replace('\\', '/')
            fps, w, h = fps_map.get(rel, (30.0, 0, 0))
            sq = Seq(root, dp, [os.path.join(dp, f) for f in exr], fps, team, prefix)
            sq.w, sq.h = w, h
            seqs.append(sq)
    seqs.sort(key=lambda s: s.id)
    return seqs


def _cache_path(seq):
    sig = seq.id + str(seq.n) + str(sum(int(os.path.getmtime(f)) for f in seq.files[:3]))
    h = hashlib.md5(sig.encode()).hexdigest()[:16]
    d = os.path.join(seq.root, '.cache')
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, h + '.npy')


def _flat(fr, team):
    """remplace les couleurs par une teinte unie (avec un leger degrade vertical), garde l'alpha."""
    out = fr.copy()
    a = out[..., 3:4].astype(np.float32) / 255
    g = np.linspace(1.18, 0.78, out.shape[-3], dtype=np.float32).reshape(-1, 1, 1)
    col = np.array(TEAM_COL[team], np.float32)
    out[..., :3] = np.clip(col * g * a, 0, 255).astype(np.uint8)
    return out


def _set(seq, arr):
    seq.frames = _flat(arr, seq.team) if ANON else arr


def load(seq):
    return _load(seq)


def _load(seq):
    """Charge toute la sequence en memoire (recadree sur l'union des pixels visibles, alpha pre-multiplie)."""
    with seq.lock:
        if seq.frames is not None:
            return seq.frames
        cp = _cache_path(seq)
        if os.path.exists(cp):
            try:
                _set(seq, np.load(cp))
                return seq.frames
            except Exception:
                pass
        ims = [_read_exr(f) for f in seq.files]
        a = np.max([im[..., 3] for im in ims], axis=0)
        ys, xs = np.where(a > 0.02)
        if len(ys) == 0:
            y0, y1, x0, x1 = 0, ims[0].shape[0], 0, ims[0].shape[1]
        else:
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        out = np.zeros((len(ims), y1 - y0, x1 - x0, 4), np.uint8)
        for i, im in enumerate(ims):
            c = np.clip(im[y0:y1, x0:x1], 0, 1)
            c[..., :3] *= c[..., 3:4]
            out[i] = (c * 255 + 0.5).astype(np.uint8)
        if out.shape[1] > MAX_H:
            sc = MAX_H / out.shape[1]
            nw = max(1, int(out.shape[2] * sc))
            out = np.stack([cv2.resize(f, (nw, MAX_H), interpolation=cv2.INTER_AREA) for f in out])
        np.save(cp, out)
        _set(seq, out)
        return seq.frames


def unload(seq):
    if seq.virtual is None:
        seq.frames = None


def thumbnail(seq, h=84):
    """Vignette RGB sur fond blanc (image du milieu), sans charger toute la sequence. Mise en cache sur disque."""
    if seq.thumb is not None:
        return seq.thumb
    if seq.frames is not None:
        fr = seq.frames[seq.n // 2].astype(np.float32)
        rgb = fr[..., :3] + 255 * (1 - fr[..., 3:4] / 255)
    else:
        tp = None
        if ANON:
            im = np.clip(_read_exr(seq.files[seq.n // 2]), 0, 1)
            a = im[..., 3:4]
            rgb = (np.array(TEAM_COL[seq.team], np.float32) / 255 * a + (1 - a)) * 255
            seq.thumb = None
        elif seq.files:
            tp = os.path.join(seq.root, '.cache', 'th_' + hashlib.md5((seq.id + str(seq.n)).encode()).hexdigest()[:16] + '.png')
            if os.path.exists(tp):
                im = cv2.imread(tp, cv2.IMREAD_COLOR)
                if im is not None:
                    seq.thumb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                    return seq.thumb
        if not ANON:
            im = np.clip(_read_exr(seq.files[seq.n // 2]), 0, 1)
            rgb = (im[..., :3] * im[..., 3:4] + (1 - im[..., 3:4])) * 255
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    s = h / rgb.shape[0]
    seq.thumb = cv2.resize(rgb, (max(1, int(rgb.shape[1] * s)), h), interpolation=cv2.INTER_AREA)
    if seq.frames is None and seq.files and tp:
        try:
            os.makedirs(os.path.dirname(tp), exist_ok=True)
            cv2.imwrite(tp, cv2.cvtColor(seq.thumb, cv2.COLOR_RGB2BGR))
        except Exception:
            pass
    return seq.thumb


LOOP_MODES = ['Coupe nette', 'Ping-pong', 'Fondu enchaine', 'Inverse']


def make_loop(src, start, n, mode, xf=3, team=None):
    """Cree une boucle virtuelle de n images a partir de src (indices 0-based), meme imparfaite.
    Coupe nette : saut possible au bouclage. Ping-pong : aller-retour. Fondu enchaine : les xf dernieres images
    se fondent vers les premieres. Inverse : lecture a l'envers."""
    fr = load(src)
    start = max(0, min(start, src.n - 1))
    n = max(2, min(n, src.n - start))
    base = fr[start:start + n]
    if mode == 'Ping-pong':
        out = np.concatenate([base, base[-2:0:-1]]) if n > 2 else base
    elif mode == 'Inverse':
        out = base[::-1]
    elif mode == 'Fondu enchaine':
        out = base.copy()
        xf = max(1, min(xf, n // 2))
        for i in range(xf):
            w = (i + 1) / (xf + 1)
            a = base[n - xf + i].astype(np.float32)
            b = base[i].astype(np.float32)
            out[n - xf + i] = np.clip(a * (1 - w) + b * w, 0, 255).astype(np.uint8)
    else:
        out = base
    s = Seq(src.root, src.path, [], src.fps, src.team if team is None else team)
    s.id = f'{src.id}#{start + 1}+{n}:{mode[:4]}' + (f'x{xf}' if mode == 'Fondu enchaine' else '')
    s.name = f'{src.name} [{start + 1}-{start + n} {mode}]'
    s.clip = src.clip
    s.n = len(out)
    s.frames = np.ascontiguousarray(out)
    s.virtual = dict(src=src.id, start=start, n=n, mode=mode, xf=xf, team=s.team)
    return s
