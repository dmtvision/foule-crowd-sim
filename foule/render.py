"""Rendu d'une image de foule : sprites detoures, perspective, ombres, fond, export video."""
import collections
import os
import subprocess

import cv2
import numpy as np

from . import lib as L
from . import sim as S


class SpriteCache:
    """Cache LRU (limite en octets) des images redimensionnees / retournees."""

    def __init__(self, max_bytes=2_500_000_000):
        self.d = collections.OrderedDict()
        self.bytes = 0
        self.max = max_bytes

    def get(self, key, make):
        v = self.d.get(key)
        if v is not None:
            self.d.move_to_end(key)
            return v
        v = make()
        self.d[key] = v
        self.bytes += v.nbytes
        while self.bytes > self.max and len(self.d) > 1:
            _, old = self.d.popitem(last=False)
            self.bytes -= old.nbytes
        return v


def hex_rgb(s):
    s = s.lstrip('#')
    return np.array([int(s[i:i + 2], 16) for i in (0, 2, 4)], np.uint8)


def over(canvas, sp, x, y):
    h, w = sp.shape[:2]
    H, W = canvas.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, W), min(y + h, H)
    if x1 <= x0 or y1 <= y0:
        return
    s = sp[y0 - y:y1 - y, x0 - x:x1 - x]
    d = canvas[y0:y1, x0:x1]
    t = d.astype(np.uint16)
    t *= (255 - s[..., 3:4]).astype(np.uint16)
    t += 127
    t //= 255
    t += s
    np.minimum(t, 255, out=t)
    d[...] = t


class Renderer:
    def __init__(self, seqs_by_id):
        self.seqs = seqs_by_id
        self.cache = SpriteCache()
        self.bgimg = {}

    def sprite(self, sid, idx, th, flip):
        s = self.seqs[sid]
        fr = s.frames
        def make():
            a = fr[idx]
            sc = th / a.shape[0]
            w = max(1, int(round(a.shape[1] * sc)))
            out = cv2.resize(a, (w, th), interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_LINEAR)
            return np.ascontiguousarray(out[:, ::-1]) if flip else out
        return self.cache.get((sid, idx, th, flip), make)

    def background(self, P, w, h):
        bg = np.empty((h, w, 3), np.uint8)
        bg[...] = hex_rgb(P['bg'])
        path = P.get('bg_image') or ''
        if path and os.path.exists(path):
            k = (path, w, h)
            if k not in self.bgimg:
                im = cv2.imread(path, cv2.IMREAD_COLOR)
                if im is not None:
                    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                    s = max(w / im.shape[1], h / im.shape[0])
                    im = cv2.resize(im, (int(im.shape[1] * s) + 1, int(im.shape[0] * s) + 1))
                    self.bgimg[k] = im[:h, :w]
            if k in self.bgimg:
                bg = self.bgimg[k].copy()
        return bg

    def render(self, crowd, P, k, transparent=False):
        """-> RGB uint8 (ou RGBA droit si transparent). k = echelle canvas -> pixels de sortie."""
        W, H = crowd.W, crowd.H
        w, h = int(round(W * k)), int(round(H * k))
        canvas = np.zeros((h, w, 4), np.uint8)
        n = crowd.n
        ids = crowd.seq_ids
        if n and ids:
            pos = crowd.disp
            ysc = np.clip(pos[:, 1] / H, 0, 1.2)
            scale = P['smin'] + (P['smax'] - P['smin']) * ysc ** P['gamma']
            th_all = P['size_pct'] / 100 * H * k * scale * crowd.scale_rand * crowd.size_mult
            ok = crowd.alive & crowd.visible
            order = [i for i in np.argsort(pos[:, 1]) if ok[i]]
            # indices d'image
            idx = np.zeros(n, int)
            for si, sid in enumerate(ids):
                m = (crowd.seqs == si) & ok
                if m.any():
                    s = self.seqs[sid]
                    idx[m] = S.frame_index(P['anim_mode'], crowd.phase[m], s.n, s.fps, P['cadence'], crowd.rate_rand[m])
            if P['shadow'] > 0:
                sh = np.zeros((h, w), np.uint8)
                for i in order:
                    th = th_all[i]
                    if th < 6:
                        continue
                    cv2.ellipse(sh, (int(pos[i, 0] * k), int(pos[i, 1] * k)), (max(2, int(th * 0.16)), max(1, int(th * 0.045))),
                                0, 0, 360, 255, -1, cv2.LINE_AA)
                sig = max(1.0, 0.012 * h)
                sh = cv2.GaussianBlur(sh, (0, 0), sig)
                canvas[..., 3] = (sh.astype(np.float32) * P['shadow']).astype(np.uint8)
            flipdir = P['flip_dir']
            vx = crowd.vel[:, 0]
            for i in order:
                th = int(round(th_all[i] / 3.0) * 3) if th_all[i] > 30 else int(round(th_all[i]))
                if th < 6:
                    continue
                sid = ids[crowd.seqs[i]]
                s = self.seqs[sid]
                if s.frames is None:
                    continue
                flip = bool(crowd.flip_rand[i]) if not flipdir else bool(vx[i] < 0)
                if s.id in P.get('flip_ids', ()):
                    flip = not flip
                sp = self.sprite(sid, int(idx[i]) % s.n, th, flip)
                x = int(pos[i, 0] * k - sp.shape[1] / 2)
                y = int(pos[i, 1] * k - sp.shape[0] * 0.97)
                over(canvas, sp, x, y)
        if transparent:
            a = canvas[..., 3:4].astype(np.float32)
            rgb = np.where(a > 0, canvas[..., :3].astype(np.float32) * 255 / np.maximum(a, 1), 0)
            return np.dstack([np.clip(rgb, 0, 255), a]).astype(np.uint8)
        bg = self.background(P, w, h)
        a = canvas[..., 3:4].astype(np.uint16)
        out = canvas[..., :3].astype(np.uint16) + (bg.astype(np.uint16) * (255 - a) + 127) // 255
        return np.minimum(out, 255).astype(np.uint8)


# ---------------------------------------------------------------- export
FORMATS = {
    'MP4 H.264 (qualite haute)': dict(ext='mp4', alpha=False,
                                       args=['-c:v', 'libx264', '-preset', 'medium', '-crf', '16', '-pix_fmt', 'yuv420p', '-movflags', '+faststart']),
    'MP4 H.264 web (leger)': dict(ext='mp4', alpha=False,
                                   args=['-c:v', 'libx264', '-preset', 'medium', '-crf', '27', '-pix_fmt', 'yuv420p', '-movflags', '+faststart']),
    'MP4 H.265 (HEVC)': dict(ext='mp4', alpha=False,
                              args=['-c:v', 'libx265', '-crf', '20', '-pix_fmt', 'yuv420p', '-tag:v', 'hvc1', '-movflags', '+faststart']),
    'WebM VP9 transparent': dict(ext='webm', alpha=True,
                                  args=['-c:v', 'libvpx-vp9', '-pix_fmt', 'yuva420p', '-b:v', '0', '-crf', '30', '-auto-alt-ref', '0']),
    'ProRes 4444 transparent (MOV)': dict(ext='mov', alpha=True,
                                           args=['-c:v', 'prores_ks', '-profile:v', '4444', '-pix_fmt', 'yuva444p10le']),
    'ProRes 422 HQ (MOV)': dict(ext='mov', alpha=False,
                                 args=['-c:v', 'prores_ks', '-profile:v', '3', '-pix_fmt', 'yuv422p10le']),
    'Sequence PNG transparente': dict(ext='png', alpha=True, args=None),
    'GIF (web)': dict(ext='gif', alpha=False, args='gif'),
}
RESOLUTIONS = {'Taille du canvas': 0, '4K UHD (3840 px de large)': 3840, '2K (2560 px)': 2560,
               '1080p (1920 px)': 1920, '720p web (1280 px)': 1280, '480p (854 px)': 854}


def export_video(crowd_factory, rend, P, out_path, fmt, width, fps, seconds, progress=None, cancel=None):
    """crowd_factory() -> Crowd neuf (meme graine). Rend a width px de large."""
    f = FORMATS[fmt]
    crowd = crowd_factory()
    W, H = crowd.W, crowd.H
    ow = width if width else W
    oh = int(round(ow * H / W))
    ow -= ow % 2
    oh -= oh % 2
    k = ow / W
    total = int(round(seconds * fps))
    dt = 1.0 / fps
    for _ in range(int(P.get('warmup', 0) * fps)):
        crowd.step(dt, P)
    alpha = f['alpha']
    proc = None
    folder = None
    if f['ext'] == 'png':
        folder = os.path.splitext(out_path)[0]
        os.makedirs(folder, exist_ok=True)
    else:
        cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgba' if alpha else 'rgb24',
               '-s', f'{ow}x{oh}', '-r', str(fps), '-i', '-']
        if f['args'] == 'gif':
            cmd += ['-vf', 'split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4', '-loop', '0']
        else:
            cmd += f['args']
        cmd += [out_path]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for i in range(total):
            if cancel and cancel():
                break
            crowd.step(dt, P)
            img = rend.render(crowd, P, k, transparent=alpha)
            if folder:
                cv2.imwrite(os.path.join(folder, f'frame_{i + 1:05d}.png'),
                            cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA))
            else:
                proc.stdin.write(np.ascontiguousarray(img).tobytes())
            if progress:
                progress(i + 1, total)
    finally:
        if proc:
            try:
                proc.stdin.close()
            except Exception:
                pass
            err = proc.stderr.read().decode(errors='replace')
            proc.wait()
            if proc.returncode not in (0, None) and not (cancel and cancel()):
                raise RuntimeError('ffmpeg : ' + err[-600:])
    return folder or out_path
