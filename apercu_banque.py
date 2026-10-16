"""Planche d'apercu : 3 images (debut, milieu, fin) de chaque sequence de banque\, composees sur fond magenta.
usage: python apercu_banque.py <dossier_clip_dans_banque> <sortie.jpg>
"""
import glob
import os
import sys

import cv2
import numpy as np
import OpenEXR


def lire(p):
    rgba = np.array(OpenEXR.File(p).channels()['RGBA'].pixels, dtype=np.float32)
    return rgba


def vignette(p, h=220):
    a = lire(p)
    bg = np.zeros_like(a[..., :3]) + np.array([1.0, 0.0, 1.0], np.float32)
    out = a[..., :3] * a[..., 3:4] + bg * (1 - a[..., 3:4])
    out = (np.clip(out, 0, 1) * 255).astype(np.uint8)
    s = h / out.shape[0]
    out = cv2.resize(out, (max(1, int(out.shape[1] * s)), h))
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


rows = []
for d in sorted(glob.glob(os.path.join(sys.argv[1], 'p*_s*'))):
    fs = sorted(glob.glob(os.path.join(d, '*.exr')))
    if not fs:
        continue
    pick = [fs[0], fs[len(fs) // 2], fs[-1]]
    ims = [vignette(p) for p in pick]
    row = np.hstack([np.pad(i, ((0, 0), (0, 6), (0, 0)), constant_values=255) for i in ims])
    cv2.putText(row, os.path.basename(d) + f' ({len(fs)})', (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    rows.append(row)
W = max(r.shape[1] for r in rows)
cols = 2
cells = [np.pad(r, ((0, 6), (0, W - r.shape[1]), (0, 0)), constant_values=255) for r in rows]
if len(cells) % cols:
    cells.append(np.full_like(cells[0], 255))
sheet = np.vstack([np.hstack(cells[i:i + cols]) for i in range(0, len(cells), cols)])
cv2.imwrite(sys.argv[2], sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
print(sheet.shape)
