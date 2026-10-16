"""Re-classe les sequences deja detourees (CLIP sur 3 images detourees) et ecarte les intrus.

usage: python trier_banque.py banque_manifestants --garde civil [--seuil 0.6] [--dry] [--clip NOM]
       python trier_banque.py banque --garde police

--garde civil  : garde les civils (ecarte policiers, journalistes, ambigus)
--garde police : garde les policiers
Les sequences ecartees sont DEPLACEES (rien n'est supprime) dans <banque>\\.ecartes\\<clip>\\<sequence>
(le dossier .ecartes est ignore par l'app ; remets-les a la main pour annuler).
Les scores sont ecrits dans <banque>\\tri.csv.
"""
import argparse
import csv
import os
import shutil

import numpy as np
import OpenEXR
import torch
from transformers import CLIPModel, CLIPProcessor

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
PROMPTS = [
    'a riot police officer wearing a helmet, dark uniform and protective gear',        # 0 police
    'a gendarme or police officer in uniform',                                          # 1 police
    'a protester in everyday civilian clothes, no helmet, no uniform',                  # 2 civil
    'a young person in casual clothes, hoodie or jacket and jeans',                     # 3 civil
    'a journalist or photographer holding a camera',                                    # 4 presse
    'a person wearing a high-visibility vest or a medic vest',                          # 5 presse/secours
]
POLICE, CIVIL = (0, 1), (2, 3)


def sprite_rgb(path):
    im = np.array(OpenEXR.File(path).channels()['RGBA'].pixels, dtype=np.float32)
    a = np.clip(im[..., 3:4], 0, 1)
    rgb = np.clip(im[..., :3], 0, 1) * a + 0.5 * (1 - a)
    return (rgb * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('banque')
    ap.add_argument('--garde', choices=['civil', 'police'], required=True)
    ap.add_argument('--seuil', type=float, default=0.5, help='probabilite mini de la classe gardee')
    ap.add_argument('--min-h', type=int, default=110, help='hauteur mini du recadrage (px)')
    ap.add_argument('--min-images', type=int, default=8)
    ap.add_argument('--dry', action='store_true', help='calcule tri.csv sans rien deplacer')
    ap.add_argument('--clip', default=None, help='ne traiter que ce dossier de clip')
    a = ap.parse_args()

    root = os.path.abspath(a.banque)
    model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(DEV).eval()
    proc = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    with torch.no_grad():
        ti = proc(text=PROMPTS, return_tensors='pt', padding=True).to(DEV)
        tf = model.get_text_features(**ti)
        tf = tf / tf.norm(dim=-1, keepdim=True)

    seqs = []
    for dp, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        ex = sorted(f for f in files if f.lower().endswith('.exr'))
        if ex:
            seqs.append((dp, ex))
    if a.clip:
        seqs = [s for s in seqs if os.path.relpath(s[0], root).replace('\\', '/').startswith(a.clip)]
    seqs.sort()
    print(len(seqs), 'sequences a classer sur', DEV, flush=True)

    rows, moved = [], 0
    B = 24
    for k0 in range(0, len(seqs), B):
        batch = seqs[k0:k0 + B]
        imgs, owner = [], []
        for si, (dp, ex) in enumerate(batch):
            n = len(ex)
            for f in (0.2, 0.5, 0.8):
                try:
                    imgs.append(sprite_rgb(os.path.join(dp, ex[min(n - 1, int(n * f))])))
                    owner.append(si)
                except Exception:
                    pass
        if not imgs:
            continue
        with torch.no_grad():
            px = proc(images=imgs, return_tensors='pt').to(DEV)
            ie = model.get_image_features(**px)
            ie = ie / ie.norm(dim=-1, keepdim=True)
            pr = (100 * ie @ tf.T).softmax(-1).cpu().numpy()
        owner = np.array(owner)
        for si, (dp, ex) in enumerate(batch):
            m = owner == si
            if not m.any():
                continue
            p = pr[m].mean(0)
            police = float(p[list(POLICE)].sum())
            civil = float(p[list(CIVIL)].sum())
            presse = float(p[4] + p[5])
            im0 = sprite_rgb(os.path.join(dp, ex[len(ex) // 2]))
            h, w = im0.shape[:2]
            score = civil if a.garde == 'civil' else police
            keep = score >= a.seuil and (police < 0.25 if a.garde == 'civil' else civil < 0.3) and h >= a.min_h and len(ex) >= a.min_images
            rel = os.path.relpath(dp, root)
            rows.append([rel.replace('\\', '/'), len(ex), w, h, round(police, 3), round(civil, 3), round(presse, 3), int(keep)])
            if not keep and not a.dry:
                dest = os.path.join(root, '.ecartes', rel)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.move(dp, dest)
                moved += 1
        print(f'{min(k0 + B, len(seqs))}/{len(seqs)}  ecartees {sum(1 for r in rows if not r[-1])}', flush=True)

    out = os.path.join(root, 'tri.csv') if not a.clip else os.path.join(root, 'tri_%s.csv' % a.clip[:30])
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['dossier', 'images', 'largeur', 'hauteur', 'p_police', 'p_civil', 'p_presse', 'garde'])
        w.writerows(rows)
    print('gardees', sum(r[-1] for r in rows), 'ecartees', sum(1 for r in rows if not r[-1]), 'deplacees', moved)


if __name__ == '__main__':
    main()
