"""Detoure chaque policier (CRS) vu en entier dans les clips de brut\ -> banque\ (EXR DWAA, fond transparent).

usage: python detourer.py [--clips a.mp4 b.mp4 ...] [--analyse]   (sans --clips : tous les mp4 de brut\)
  --analyse : detection + suivi + classement police/civil, affiche les pistes, n'ecrit rien.

Sortie : banque\<clip>\p<id>_s<nn>\<clip>_p<id>_s<nn>_<frame>.exr   (RGBA float16, couleurs sRGB, alpha droit)
         banque\index.csv  (une ligne par sequence)
"""
import argparse
import csv
import glob
import os
import sys

import cv2
import numpy as np
import torch
import OpenEXR
from ultralytics import SAM, YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
BRUT = os.path.join(HERE, 'brut')
BANQUE = os.path.join(HERE, 'banque')
MIN_H = 90            # hauteur mini du personnage en pixels
MARGE = 0.015         # marge au bord de l'image (fraction)
MAX_OCCL = 0.20       # part de la boite recouverte par une autre personne
MIN_RATIO = 1.5       # hauteur/largeur mini (debout, en entier)
MIN_SEQ = 8           # images mini par sequence
MAX_GAP = 2
DEV = 'cuda'


def track_clip(path, yolo):
    """-> frames (liste de dict id -> (box, conf)), fps, (w, h)"""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(3)), int(cap.get(4))
    cap.release()
    frames = []
    for r in yolo.track(path, stream=True, persist=True, classes=[0], conf=0.25, iou=0.6,
                        tracker='botsort.yaml', imgsz=1280, device=DEV, verbose=False):
        d = {}
        if r.boxes is not None and r.boxes.id is not None:
            for b, i, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.id.int().cpu().tolist(), r.boxes.conf.cpu().tolist()):
                d[i] = (b, c)
        frames.append(d)
    return frames, fps, (w, h)


def police_scores(path, frames, clip_model, clip_proc):
    """probabilite 'policier en tenue de maintien de l'ordre' par piste (moyenne sur ~8 crops)."""
    prompts = ['a riot police officer in dark uniform with helmet and protective gear',
               'a civilian protester or student in everyday clothes',
               'a journalist or photographer']
    samples = {}
    for fi, d in enumerate(frames):
        for i, (b, c) in d.items():
            samples.setdefault(i, []).append((fi, b, c))
    want = {}
    for i, s in samples.items():
        s = sorted(s, key=lambda t: -t[2] * (t[1][3] - t[1][1]))[:8]
        for fi, b, c in s:
            want.setdefault(fi, []).append((i, b))
    cap = cv2.VideoCapture(path)
    crops = {i: [] for i in samples}
    fi = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        for i, b in want.get(fi, []):
            x1, y1, x2, y2 = [int(v) for v in b]
            c = fr[max(y1, 0):y2, max(x1, 0):x2]
            if c.size:
                crops[i].append(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
        fi += 1
    cap.release()
    out = {}
    for i, cs in crops.items():
        if not cs:
            out[i] = (0.0, 0.0, 0.0)
            continue
        inp = clip_proc(text=prompts, images=cs, return_tensors='pt', padding=True).to(DEV)
        with torch.no_grad():
            p = clip_model(**inp).logits_per_image.softmax(-1).mean(0).tolist()
        out[i] = tuple(p)
    return out


def good_frames(frames, ident, size):
    """images ou le personnage est entier, debout et peu cache -> liste bool"""
    w, h = size
    ok = []
    for d in frames:
        if ident not in d:
            ok.append(False)
            continue
        b, c = d[ident]
        x1, y1, x2, y2 = b
        bw, bh = x2 - x1, y2 - y1
        good = (bh >= MIN_H and bh / max(bw, 1) >= MIN_RATIO and c >= 0.4 and x1 > MARGE * w and y1 > MARGE * h
                and x2 < (1 - MARGE) * w and y2 < (1 - MARGE) * h)
        if good:
            for j, (ob, oc) in d.items():
                if j == ident:
                    continue
                iw = min(x2, ob[2]) - max(x1, ob[0])
                ih = min(y2, ob[3]) - max(y1, ob[1])
                if iw > 0 and ih > 0 and iw * ih / (bw * bh) > MAX_OCCL:
                    good = False
                    break
        ok.append(good)
    return ok


def segments(ok):
    segs, start, gap = [], None, 0
    for i, g in enumerate(ok):
        if g:
            if start is None:
                start = i
            last, gap = i, 0
        elif start is not None:
            gap += 1
            if gap > MAX_GAP:
                segs.append((start, last))
                start = None
    if start is not None:
        segs.append((start, last))
    return [(a, b) for a, b in segs if b - a + 1 >= MIN_SEQ]


def write_exr(path, rgb, alpha):
    rgba = np.dstack([rgb.astype(np.float32) / 255.0, alpha.astype(np.float32) / 255.0]).astype(np.float16)
    OpenEXR.File({'compression': OpenEXR.DWAA_COMPRESSION, 'type': OpenEXR.scanlineimage}, {'RGBA': rgba}).write(path)


def export(path, name, frames, fps, size, jobs, sam, writer, banque):
    """jobs: liste (ident, a, b). Une passe de lecture, SAM2 par image avec boites en prompt."""
    w, h = size
    byframe = {}
    plan = []
    for ident, a, b in jobs:
        n = len([p for p in plan if p['id'] == ident]) + 1
        box = np.array([frames[f][ident][0] for f in range(a, b + 1) if ident in frames[f]])
        x1, y1, x2, y2 = box[:, 0].min(), box[:, 1].min(), box[:, 2].max(), box[:, 3].max()
        px, py = 0.08 * (x2 - x1), 0.06 * (y2 - y1)
        x1, y1 = int(max(0, x1 - px)), int(max(0, y1 - py))
        x2, y2 = int(min(w, x2 + px)), int(min(h, y2 + py))
        x2 -= (x2 - x1) % 2
        y2 -= (y2 - y1) % 2
        p = dict(id=ident, a=a, b=b, seq=n, crop=(x1, y1, x2, y2), dir=os.path.join(banque, name, f'p{ident:02d}_s{n:02d}'))
        os.makedirs(p['dir'], exist_ok=True)
        plan.append(p)
        for f in range(a, b + 1):
            if ident in frames[f]:
                byframe.setdefault(f, []).append(p)
    cap = cv2.VideoCapture(path)
    fi = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        for p in byframe.get(fi, []):
            pass
        if fi in byframe:
            ps = byframe[fi]
            boxes = [frames[fi][p['id']][0].tolist() for p in ps]
            res = sam(fr, bboxes=boxes, verbose=False, device=DEV)
            masks = res[0].masks.data.cpu().numpy() if res[0].masks is not None else None
            if masks is not None and len(masks) == len(ps):
                for p, m in zip(ps, masks):
                    m = cv2.resize(m.astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_LINEAR) if m.shape != (h, w) else m.astype(np.uint8) * 255
                    # garde la composante qui recouvre le mieux la boite (retire les voisins detaches)
                    n, lab, st, _ = cv2.connectedComponentsWithStats((m > 127).astype(np.uint8), connectivity=8)
                    if n > 2:
                        bx = frames[fi][p['id']][0].astype(int)
                        cx, cy = (bx[0] + bx[2]) // 2, (bx[1] + bx[3]) // 2
                        keep = lab[min(max(cy, 0), h - 1), min(max(cx, 0), w - 1)]
                        if keep == 0:
                            keep = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
                        m = np.where(lab == keep, m, 0).astype(np.uint8)
                    m = cv2.erode(m, np.ones((3, 3), np.uint8))
                    m = cv2.GaussianBlur(m, (0, 0), 0.9)
                    x1, y1, x2, y2 = p['crop']
                    rgb = cv2.cvtColor(fr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
                    write_exr(os.path.join(p['dir'], f"{name}_p{p['id']:02d}_s{p['seq']:02d}_{fi - p['a'] + 1:05d}.exr"),
                              rgb, m[y1:y2, x1:x2])
        fi += 1
    cap.release()
    for p in plan:
        x1, y1, x2, y2 = p['crop']
        writer.writerow([name, os.path.relpath(p['dir'], banque), p['b'] - p['a'] + 1, round(fps, 3), x2 - x1, y2 - y1,
                         p['a'] + 1, p['b'] + 1, p['crop'][0], p['crop'][1]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clips', nargs='*')
    ap.add_argument('--analyse', action='store_true')
    ap.add_argument('--classes', default='police', help='police, manif ou police,manif')
    ap.add_argument('--seuil', type=float, default=0.6, help='proba mini police')
    ap.add_argument('--seuil-manif', type=float, default=0.55, help='proba mini civil')
    ap.add_argument('--sam', default='sam2.1_b.pt')
    args = ap.parse_args()
    classes = [c.strip() for c in args.classes.split(',')]
    roots = {'police': BANQUE, 'manif': os.path.join(HERE, 'banque_manifestants')}
    clips = [os.path.join(BRUT, c) if not os.path.isabs(c) else c for c in args.clips] if args.clips else         sorted(p for p in glob.glob(os.path.join(BRUT, '*.mp4')) if '.fhls' not in p)
    from transformers import CLIPModel, CLIPProcessor
    clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(DEV).eval()
    clip_proc = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    yolo = YOLO('yolo11x.pt')
    sam = None if args.analyse else SAM(args.sam)
    files, writers = {}, {}
    for c in classes:
        os.makedirs(roots[c], exist_ok=True)
        ip = os.path.join(roots[c], 'index.csv')
        new = not os.path.exists(ip)
        files[c] = open(ip, 'a', newline='', encoding='utf-8')
        writers[c] = csv.writer(files[c])
        if new:
            writers[c].writerow(['clip', 'dossier', 'images', 'fps', 'largeur', 'hauteur', 'image_debut', 'image_fin', 'crop_x', 'crop_y'])
    done = {}
    for c in classes:
        dp = os.path.join(roots[c], 'traites.txt')
        done[c] = set(open(dp, encoding='utf-8').read().split()) if os.path.exists(dp) else set()
    for path in clips:
        name = os.path.splitext(os.path.basename(path))[0]
        todo = [c for c in classes if name not in done[c]]
        if not todo:
            continue
        yolo.predictor = None
        frames, fps, size = track_clip(path, yolo)
        score = police_scores(path, frames, clip_model, clip_proc)
        for c in todo:
            jobs = []
            for ident, pr in sorted(score.items()):
                ok = good_frames(frames, ident, size)
                segs = segments(ok)
                want = pr[0] >= args.seuil if c == 'police' else pr[1] >= args.seuil_manif
                if want:
                    jobs += [(ident, a, b) for a, b in segs]
            print(f'== {name} [{c}]: {len(jobs)} sequences', flush=True)
            if not args.analyse:
                if jobs:
                    export(path, name, frames, fps, size, jobs, sam, writers[c], roots[c])
                    files[c].flush()
                with open(os.path.join(roots[c], 'traites.txt'), 'a', encoding='utf-8') as f:
                    f.write(name + '\n')


if __name__ == '__main__':
    main()
