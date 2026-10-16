"""Foule - foules animees (CRS + manifestants qui interagissent) a partir de sequences detourees.
Lancer :  python -m foule   (depuis le dossier crs-vids)  ou  lancer_foule.bat"""
import copy
import json
import os
import sys
import time

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets as W

from . import lib as L
from . import render as R
from . import sim as S

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SETTINGS = os.path.join(HERE, 'reglages.json')

CANVAS_PRESETS = {'16:9  1920x1080': (1920, 1080), '16:9  4K 3840x2160': (3840, 2160), '9:16  vertical 1080x1920': (1080, 1920),
                  '1:1  1080x1080': (1080, 1080), '4:3  1440x1080': (1440, 1080), '21:9  2560x1080': (2560, 1080),
                  'Personnalise': None}
TEAM_BG = [QtGui.QColor(215, 228, 250), QtGui.QColor(255, 232, 205)]
TEAM_TAG = ['CRS', 'MANIF']


def default_P():
    P = dict(
        folder_crs=os.path.join(ROOT, 'banque'), folder_manif=os.path.join(ROOT, 'banque_manifestants'),
        scenario='Libre (aucune interaction)', seed=1,
        teams=[S.team_defaults(0), S.team_defaults(1)], inter=dict(ab=S.inter_defaults(), ba=S.inter_defaults()),
        cap_ab='Rien', cap_ba='Rien', cap_radius=3.0, cap_hold=2.0,
        anim_mode=S.ANIM_MODES[0], anim_speed=1.0, cadence=8, sync_speed=True, pos_stepped=False, boil=0.0, flip_dir=True,
        canvas='16:9  1920x1080', cw=1920, ch=1080, bg='#ffffff', bg_image='', transparent=False, size_pct=24, smin=0.55,
        smax=1.25, gamma=1.0, shadow=0.0,
        fmt='MP4 H.264 (qualite haute)', res='1080p (1920 px)', fps=25, seconds=10.0, warmup=0.0,
        out_dir=os.path.join(ROOT, 'rendus'), out_name='foule', min_frames=8,
        loops=[], team_override={}, flip_ids=[],
    )
    P['teams'][0]['n'] = 60
    P['teams'][1]['n'] = 40
    return P


# ------------------------------------------------------------------ petits widgets
class Slider(W.QWidget):
    changed = QtCore.Signal(object)

    def __init__(self, label, lo, hi, val, step=1.0, dec=0, suffix=''):
        super().__init__()
        self.step, self.dec, self.suffix = step, dec, suffix
        lay = W.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.lab = W.QLabel(label)
        self.lab.setMinimumWidth(130)
        self.sl = W.QSlider(QtCore.Qt.Horizontal)
        self.sl.setRange(int(round(lo / step)), int(round(hi / step)))
        self.val = W.QLabel()
        self.val.setMinimumWidth(54)
        self.val.setAlignment(QtCore.Qt.AlignRight)
        lay.addWidget(self.lab)
        lay.addWidget(self.sl, 1)
        lay.addWidget(self.val)
        self.set(val)
        self.sl.valueChanged.connect(self._ch)

    def value(self):
        v = self.sl.value() * self.step
        return int(round(v)) if self.dec == 0 else round(v, self.dec)

    def set(self, v):
        self.sl.blockSignals(True)
        self.sl.setValue(int(round(v / self.step)))
        self.sl.blockSignals(False)
        self._show()

    def _show(self):
        self.val.setText(f'{self.value():.{self.dec}f}{self.suffix}')

    def _ch(self, _):
        self._show()
        self.changed.emit(self.value())


class Loader(QtCore.QThread):
    progress = QtCore.Signal(int, int)
    BUDGET = 7_000_000_000      # octets de sprites gardes en memoire

    def __init__(self, seqs):
        super().__init__()
        self.seqs = seqs
        self.stop = False
        self.skipped = 0

    def run(self):
        used = 0
        for i, s in enumerate(self.seqs):
            if self.stop:
                return
            try:
                est = s.n * (s.w or 200) * (s.h or 300) * 4 if s.frames is None else s.frames.nbytes
                if used + est > self.BUDGET:
                    self.skipped += 1
                else:
                    L.load(s)
                    used += s.frames.nbytes
            except Exception as e:
                print('erreur', s.id, e)
            self.progress.emit(i + 1, len(self.seqs))


class Exporter(QtCore.QThread):
    progress = QtCore.Signal(int, int)
    done = QtCore.Signal(str)
    failed = QtCore.Signal(str)

    def __init__(self, factory, rend, P, path, fmt, width, fps, seconds):
        super().__init__()
        self.a = (factory, rend, P, path, fmt, width, fps, seconds)
        self.cancel = False

    def run(self):
        f, r, P, path, fmt, width, fps, sec = self.a
        try:
            out = R.export_video(f, r, P, path, fmt, width, fps, sec, lambda i, n: self.progress.emit(i, n), lambda: self.cancel)
            self.done.emit(out)
        except Exception as e:
            self.failed.emit(str(e))


class PreviewLabel(W.QLabel):
    clicked = QtCore.Signal(float, float)

    def __init__(self):
        super().__init__()
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setMinimumSize(480, 270)
        self.setSizePolicy(W.QSizePolicy.Expanding, W.QSizePolicy.Expanding)
        self.setStyleSheet('background:#2b2b2b;')
        self.pm = None

    def show_image(self, img):
        h, w = img.shape[:2]
        qi = QtGui.QImage(img.data, w, h, w * 3, QtGui.QImage.Format_RGB888)
        self.pm = QtGui.QPixmap.fromImage(qi)
        self.setPixmap(self.pm.scaled(self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def mousePressEvent(self, e):
        if self.pm is None:
            return
        sc = self.pm.scaled(self.size(), QtCore.Qt.KeepAspectRatio)
        ox, oy = (self.width() - sc.width()) / 2, (self.height() - sc.height()) / 2
        fx, fy = (e.position().x() - ox) / sc.width(), (e.position().y() - oy) / sc.height()
        if 0 <= fx <= 1 and 0 <= fy <= 1:
            self.clicked.emit(fx, fy)


def seq_to_pixmap(frames, i, h, flip=False):
    a = frames[i % len(frames)]
    sc = h / a.shape[0]
    im = np.ascontiguousarray(cv2.resize(a, (max(1, int(a.shape[1] * sc)), h)))
    if flip:
        im = np.ascontiguousarray(im[:, ::-1])
    rgb = np.minimum(im[..., :3].astype(np.uint16) + (255 - im[..., 3:4]), 255).astype(np.uint8)
    qi = QtGui.QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1] * 3, QtGui.QImage.Format_RGB888).copy()
    return QtGui.QPixmap.fromImage(qi)


# ------------------------------------------------------------------ createur de boucles
class LoopDialog(W.QDialog):
    def __init__(self, parent, src):
        super().__init__(parent)
        self.setWindowTitle('Creer une boucle de n images')
        self.src = src
        L.load(src)
        self.result = None
        v = W.QVBoxLayout(self)
        v.addWidget(W.QLabel(f'Source : {src.clip} / {src.name}  ({src.n} images, {src.fps:.0f} i/s)'))
        row = W.QHBoxLayout()
        self.start = W.QSpinBox()
        self.start.setRange(1, src.n)
        self.start.setPrefix('Debut : image ')
        self.n = W.QSpinBox()
        self.n.setRange(2, src.n)
        self.n.setValue(min(src.n, 12))
        self.n.setPrefix('Longueur : ')
        self.n.setSuffix(' images')
        row.addWidget(self.start)
        row.addWidget(self.n)
        v.addLayout(row)
        row = W.QHBoxLayout()
        self.mode = W.QComboBox()
        self.mode.addItems(L.LOOP_MODES)
        self.xf = W.QSpinBox()
        self.xf.setRange(1, 30)
        self.xf.setValue(3)
        self.xf.setPrefix('Fondu : ')
        self.xf.setSuffix(' images')
        self.team = W.QComboBox()
        self.team.addItems(['Equipe de la source', 'CRS', 'Manifestants'])
        row.addWidget(self.mode, 1)
        row.addWidget(self.xf)
        row.addWidget(self.team)
        v.addLayout(row)
        self.prev = W.QLabel()
        self.prev.setFixedHeight(260)
        self.prev.setAlignment(QtCore.Qt.AlignCenter)
        self.prev.setStyleSheet('background:white;border:1px solid #888;')
        v.addWidget(self.prev)
        self.info = W.QLabel()
        v.addWidget(self.info)
        v.addWidget(W.QLabel('Meme une boucle imparfaite est acceptee : "Coupe nette" ne corrige rien, "Fondu enchaine" lisse le raccord,\n'
                             '"Ping-pong" evite tout saut. La boucle apparait dans la bibliotheque comme une boucle a part.'))
        row = W.QHBoxLayout()
        ok = W.QPushButton('Ajouter la boucle')
        ok.clicked.connect(self.accept_loop)
        cancel = W.QPushButton('Fermer')
        cancel.clicked.connect(self.reject)
        row.addStretch(1)
        row.addWidget(ok)
        row.addWidget(cancel)
        v.addLayout(row)
        for w in (self.start, self.n, self.xf):
            w.valueChanged.connect(self.rebuild)
        self.mode.currentIndexChanged.connect(self.rebuild)
        self.i = 0.0
        self.t = QtCore.QTimer(self)
        self.t.timeout.connect(self.tick)
        self.rebuild()
        self.t.start(40)

    def rebuild(self, *_):
        self.start.setMaximum(max(1, self.src.n - 1))
        self.n.setMaximum(max(2, self.src.n - self.start.value() + 1))
        self.loop = L.make_loop(self.src, self.start.value() - 1, self.n.value(), self.mode.currentText(), self.xf.value())
        self.info.setText(f'Boucle : {self.loop.n} images  -  duree {self.loop.n / self.src.fps:.2f} s')
        self.i = 0.0

    def tick(self):
        self.i += self.src.fps * 0.04
        self.prev.setPixmap(seq_to_pixmap(self.loop.frames, int(self.i), 240))

    def accept_loop(self):
        t = self.team.currentIndex()
        self.result = L.make_loop(self.src, self.start.value() - 1, self.n.value(), self.mode.currentText(), self.xf.value(),
                                  None if t == 0 else t - 1)
        self.accept()


# ------------------------------------------------------------------ fenetre principale
class Main(W.QMainWindow):
    REGEN = {'seed', 'cw', 'ch'}
    TEAM_REGEN = {'n', 'place', 'zx', 'zy', 'zw', 'zh', 'rows', 'direction', 'gx', 'gy'}

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Foule - CRS et manifestants, foules animees a partir de sequences detourees')
        self.P = default_P()
        self.saved_enabled = None
        self.load_settings()
        self.seqs, self.by_id = [], {}
        self.enabled = set()
        self.crowd = None
        self.rend = R.Renderer({})
        self.widgets = {}      # reglages globaux
        self.team_widgets = {}  # reglages de l'equipe editee
        self.inter_widgets = {}
        self.cur_team = 0
        self.loader = None
        self.exporter = None
        self.playing = True
        self.fps_disp = 0.0
        self._build()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.mini_timer = QtCore.QTimer(self)
        self.mini_timer.timeout.connect(self.mini_tick)
        self.mini_i = 0.0
        self.mini_timer.start(40)
        self._items, self._thumb_queue = {}, []
        self.thumb_timer = QtCore.QTimer(self)
        self.thumb_timer.timeout.connect(self.fill_thumbs)
        self.thumb_timer.start(40)
        self.set_folder()
        self.sync_canvas_to_res()
        self.timer.start(33)
        self.resize(1600, 920)

    # ---- reglages
    def load_settings(self, path=SETTINGS):
        try:
            with open(path, encoding='utf-8') as f:
                d = json.load(f)
        except Exception:
            return
        self.saved_enabled = d.pop('_enabled', None)
        base = default_P()
        for k, v in d.items():
            if k not in base:
                continue
            if k == 'teams':
                for i, t in enumerate(v[:2]):
                    base['teams'][i].update(t)
            elif k == 'inter':
                for kk in ('ab', 'ba'):
                    base['inter'][kk].update(v.get(kk, {}))
            else:
                base[k] = v
        self.P = base

    def save_settings(self, path=SETTINGS):
        d = dict(self.P)
        d['_enabled'] = sorted(self.enabled)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(d, f, indent=1, ensure_ascii=False)

    def closeEvent(self, e):
        try:
            self.save_settings()
        except Exception:
            pass
        if self.loader:
            self.loader.stop = True
        super().closeEvent(e)

    # ---- construction de l'interface
    def _build(self):
        lib_w = W.QWidget()
        lv = W.QVBoxLayout(lib_w)
        for key, label in (('folder_crs', 'CRS'), ('folder_manif', 'Manifestants')):
            row = W.QHBoxLayout()
            e = W.QLineEdit(self.P[key])
            e.editingFinished.connect(lambda k=key, ed=e: self.change_folder(k, ed.text()))
            b = W.QPushButton('...')
            b.setMaximumWidth(34)
            b.clicked.connect(lambda _=False, k=key, ed=e: self.pick_folder(k, ed))
            row.addWidget(W.QLabel(label))
            row.addWidget(e, 1)
            row.addWidget(b)
            lv.addLayout(row)
            setattr(self, 'edit_' + key, e)
        row = W.QHBoxLayout()
        self.team_filter = W.QComboBox()
        self.team_filter.addItems(['Toutes equipes', 'CRS', 'Manifestants'])
        self.team_filter.currentIndexChanged.connect(self.refill_list)
        self.clip_combo = W.QComboBox()
        self.clip_combo.currentIndexChanged.connect(self.refill_list)
        self.min_spin = W.QSpinBox()
        self.min_spin.setRange(1, 500)
        self.min_spin.setValue(self.P['min_frames'])
        self.min_spin.setPrefix('images >= ')
        self.min_spin.valueChanged.connect(self.refill_list)
        row.addWidget(self.team_filter)
        row.addWidget(self.clip_combo, 1)
        row.addWidget(self.min_spin)
        lv.addLayout(row)
        self.list = W.QListWidget()
        self.list.setIconSize(QtCore.QSize(44, 64))
        self.list.itemChanged.connect(self.on_item_changed)
        self.list.currentItemChanged.connect(self.on_select)
        lv.addWidget(self.list, 1)
        row = W.QHBoxLayout()
        for txt, fn in [('Cocher visibles', lambda: self.check_visible(True)), ('Decocher visibles', lambda: self.check_visible(False)),
                        ('Inverser', self.invert_visible)]:
            bb = W.QPushButton(txt)
            bb.clicked.connect(fn)
            row.addWidget(bb)
        lv.addLayout(row)
        row = W.QHBoxLayout()
        b = W.QPushButton('Changer d\'equipe')
        b.clicked.connect(self.swap_team)
        self.flip_chk = W.QCheckBox('Retourner')
        self.flip_chk.toggled.connect(self.on_flip_toggle)
        row.addWidget(b)
        row.addWidget(self.flip_chk)
        lv.addLayout(row)
        row = W.QHBoxLayout()
        b1 = W.QPushButton('Creer une boucle de n images...')
        b1.clicked.connect(self.create_loop)
        b2 = W.QPushButton('Supprimer la boucle')
        b2.clicked.connect(self.delete_loop)
        row.addWidget(b1, 1)
        row.addWidget(b2)
        lv.addLayout(row)
        row = W.QHBoxLayout()
        self.auto_n = W.QSpinBox()
        self.auto_n.setRange(2, 200)
        self.auto_n.setValue(12)
        self.auto_n.setPrefix('Decoupe auto : ')
        self.auto_n.setSuffix(' img')
        b3 = W.QPushButton('Decouper les boucles cochees')
        b3.clicked.connect(self.auto_slice)
        row.addWidget(self.auto_n)
        row.addWidget(b3, 1)
        lv.addLayout(row)
        self.mini = W.QLabel()
        self.mini.setFixedHeight(190)
        self.mini.setAlignment(QtCore.Qt.AlignCenter)
        self.mini.setStyleSheet('background:white;border:1px solid #888;')
        lv.addWidget(self.mini)
        self.lib_info = W.QLabel('')
        lv.addWidget(self.lib_info)
        dock = W.QDockWidget('Bibliotheque de boucles', self)
        dock.setWidget(lib_w)
        dock.setFeatures(W.QDockWidget.NoDockWidgetFeatures)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock)
        dock.setMinimumWidth(430)

        c = W.QWidget()
        cv = W.QVBoxLayout(c)
        self.preview = PreviewLabel()
        self.preview.clicked.connect(self.on_preview_click)
        cv.addWidget(self.preview, 1)
        row = W.QHBoxLayout()
        self.play_btn = W.QPushButton('Pause')
        self.play_btn.clicked.connect(self.toggle_play)
        rs = W.QPushButton('Regenerer la foule')
        rs.clicked.connect(self.regen)
        ns = W.QPushButton('Autre graine')
        ns.clicked.connect(self.new_seed)
        cl = W.QPushButton('Retirer les points cibles')
        cl.clicked.connect(self.clear_goal)
        self.status = W.QLabel('')
        for w in (self.play_btn, rs, ns, cl):
            row.addWidget(w)
        row.addWidget(self.status, 1)
        cv.addLayout(row)
        self.setCentralWidget(c)

        right = W.QWidget()
        rv = W.QVBoxLayout(right)
        row = W.QHBoxLayout()
        row.addWidget(W.QLabel('Equipe editee (Placement / Mouvement) :'))
        self.team_combo = W.QComboBox()
        self.team_combo.addItems(S.TEAM_NAMES)
        self.team_combo.currentIndexChanged.connect(self.select_team)
        row.addWidget(self.team_combo, 1)
        rv.addLayout(row)
        tabs = W.QTabWidget()
        tabs.addTab(self._tab_placement(), 'Placement')
        tabs.addTab(self._tab_mouvement(), 'Mouvement')
        tabs.addTab(self._tab_inter(), 'Interactions')
        tabs.addTab(self._tab_animation(), 'Animation')
        tabs.addTab(self._tab_canvas(), 'Canvas / rendu')
        tabs.addTab(self._tab_export(), 'Export')
        rv.addWidget(tabs, 1)
        dock2 = W.QDockWidget('Reglages', self)
        dock2.setWidget(right)
        dock2.setFeatures(W.QDockWidget.NoDockWidgetFeatures)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock2)
        dock2.setMinimumWidth(470)
        mb = self.menuBar().addMenu('Projet')
        mb.addAction('Enregistrer le projet...', self.save_project)
        mb.addAction('Ouvrir un projet...', self.open_project)

    # helpers : reglages globaux
    def _slider(self, lay, key, label, lo, hi, step=1.0, dec=0, suffix=''):
        s = Slider(label, lo, hi, self.P[key], step, dec, suffix)
        s.changed.connect(lambda v, k=key: self.set_param(k, v))
        lay.addWidget(s)
        self.widgets[key] = s
        return s

    def _combo(self, lay, key, label, items):
        row = W.QHBoxLayout()
        row.addWidget(W.QLabel(label))
        cb = W.QComboBox()
        cb.addItems(list(items))
        cb.setCurrentText(self.P[key])
        cb.currentTextChanged.connect(lambda v, k=key: self.set_param(k, v))
        row.addWidget(cb, 1)
        lay.addLayout(row)
        self.widgets[key] = cb
        return cb

    def _check(self, lay, key, label):
        cb = W.QCheckBox(label)
        cb.setChecked(self.P[key])
        cb.toggled.connect(lambda v, k=key: self.set_param(k, v))
        lay.addWidget(cb)
        self.widgets[key] = cb
        return cb

    # helpers : reglages de l'equipe editee
    def _tslider(self, lay, key, label, lo, hi, step=1.0, dec=0, suffix=''):
        s = Slider(label, lo, hi, self.P['teams'][self.cur_team][key], step, dec, suffix)
        s.changed.connect(lambda v, k=key: self.set_team_param(k, v))
        lay.addWidget(s)
        self.team_widgets[key] = s

    def _tcombo(self, lay, key, label, items):
        row = W.QHBoxLayout()
        row.addWidget(W.QLabel(label))
        cb = W.QComboBox()
        cb.addItems(list(items))
        cb.setCurrentText(self.P['teams'][self.cur_team][key])
        cb.currentTextChanged.connect(lambda v, k=key: self.set_team_param(k, v))
        row.addWidget(cb, 1)
        lay.addLayout(row)
        self.team_widgets[key] = cb

    def _tab(self):
        w = W.QWidget()
        v = W.QVBoxLayout(w)
        return w, v

    def _tab_placement(self):
        w, v = self._tab()
        self._tcombo(v, 'place', 'Preset de placement', S.PLACEMENTS)
        self._tslider(v, 'n', 'Nombre de personnages', 0, 600, 1)
        self._tslider(v, 'zx', 'Zone : centre X', 0, 1, 0.01, 2)
        self._tslider(v, 'zy', 'Zone : centre Y (profondeur)', 0, 1, 0.01, 2)
        self._tslider(v, 'zw', 'Zone : largeur', 0.02, 1.5, 0.01, 2)
        self._tslider(v, 'zh', 'Zone : hauteur', 0.02, 1, 0.01, 2)
        self._tslider(v, 'rows', 'Rangs (lignes / colonne)', 1, 12, 1)
        self._tslider(v, 'direction', 'Direction initiale', 0, 359, 1, 0, ' deg')
        self._tslider(v, 'gx', 'Point cible X (-1 = aucun)', -1, 1, 0.01, 2)
        self._tslider(v, 'gy', 'Point cible Y', 0, 1, 0.01, 2)
        self._slider(v, 'seed', 'Graine aleatoire', 0, 999, 1)
        v.addWidget(W.QLabel('Clic dans l\'apercu : pose un point cible pour l\'equipe editee\n(attire, ou repousse si "But" est negatif).'))
        v.addStretch(1)
        return w

    def _tab_mouvement(self):
        w, v = self._tab()
        row = W.QHBoxLayout()
        row.addWidget(W.QLabel('Preset de mouvement'))
        self.move_combo = W.QComboBox()
        self.move_combo.addItems(list(S.MOVE_PRESETS))
        self.move_combo.currentTextChanged.connect(self.apply_move_preset)
        row.addWidget(self.move_combo, 1)
        v.addLayout(row)
        g = W.QGroupBox('Stabilite (maintenir en place, moins de chaos)')
        gl = W.QVBoxLayout(g)
        self._tslider(gl, 'home', "Ancrage (rappel vers l'origine)", 0, 6, 0.1, 1)
        self._tslider(gl, 'home_r', 'Rayon de liberte (% haut.)', 0, 30, 0.5, 1)
        self._tslider(gl, 'damp', 'Amortissement (frottement)', 0, 8, 0.1, 1)
        self._tslider(gl, 'smooth', 'Lissage des mouvements', 0, 1, 0.05, 2)
        self._tslider(gl, 'force', 'Force de pilotage max', 0.5, 10, 0.1, 1)
        row = W.QHBoxLayout()
        b1 = W.QPushButton('Calmer (stabiliser)')
        b1.clicked.connect(self.calm_team)
        b2 = W.QPushButton('Ancrer ici (positions actuelles)')
        b2.clicked.connect(self.anchor_now)
        row.addWidget(b1)
        row.addWidget(b2)
        gl.addLayout(row)
        v.addWidget(g)
        self._tslider(v, 'speed', 'Vitesse max (% haut. / s)', 0, 60, 0.5, 1)
        self._tslider(v, 'sep', 'Separation', 0, 5, 0.1, 1)
        self._tslider(v, 'ali', 'Alignement', 0, 5, 0.1, 1)
        self._tslider(v, 'coh', 'Cohesion', 0, 5, 0.1, 1)
        self._tslider(v, 'radius', 'Rayon de voisinage (% haut.)', 2, 60, 0.5, 1)
        self._tslider(v, 'sep_radius', 'Rayon de separation (% haut.)', 1, 30, 0.5, 1)
        self._tslider(v, 'goal', 'But (+ attire / - repousse)', -5, 5, 0.1, 1)
        self._tslider(v, 'swirl', 'Rotation (moulin)', -5, 5, 0.1, 1)
        self._tslider(v, 'wander', 'Errance (bruit)', 0, 4, 0.05, 2)
        self._tslider(v, 'wind_x', 'Courant X', -3, 3, 0.1, 1)
        self._tslider(v, 'wind_y', 'Courant Y', -3, 3, 0.1, 1)
        self._tslider(v, 'depth_mob', 'Mobilite en profondeur', 0, 1, 0.05, 2)
        self._tcombo(v, 'boundary', 'Bords du canvas', S.BOUNDARIES)
        self._tslider(v, 'size_mult', 'Taille de l\'equipe (x)', 0.4, 2.0, 0.05, 2)
        self._tslider(v, 'anim_mult', 'Vitesse d\'animation de l\'equipe (x)', 0.2, 3.0, 0.05, 2)
        v.addStretch(1)
        return w

    def _tab_inter(self):
        w, v = self._tab()
        row = W.QHBoxLayout()
        row.addWidget(W.QLabel('Strategie'))
        self.scn_combo = W.QComboBox()
        self.scn_combo.addItems(list(S.SCENARIOS))
        self.scn_combo.setCurrentText(self.P['scenario'])
        self.scn_combo.currentTextChanged.connect(self.apply_scenario)
        row.addWidget(self.scn_combo, 1)
        v.addLayout(row)
        v.addWidget(W.QLabel('Une strategie regle les deux equipes et leurs interactions ; tout reste modifiable ensuite.\n'
                             'Le "leader" d\'une equipe est son premier personnage (cible unique).'))
        for key, title in (('ab', 'CRS  ->  Manifestants'), ('ba', 'Manifestants  ->  CRS')):
            g = W.QGroupBox(title)
            gl = W.QVBoxLayout(g)
            cb = W.QComboBox()
            cb.addItems(S.PAIR_MODES)
            cb.setCurrentText(self.P['inter'][key]['mode'])
            cb.currentTextChanged.connect(lambda val, k=key: self.set_inter(k, 'mode', val))
            gl.addWidget(cb)
            self.inter_widgets[(key, 'mode')] = cb
            for field, label, lo, hi, step, dec in (('strength', 'Intensite', 0, 5, 0.1, 1), ('range', 'Portee (% haut.)', 5, 300, 1, 0),
                                                    ('lead', 'Anticipation de la cible (s)', 0, 2, 0.05, 2)):
                s = Slider(label, lo, hi, self.P['inter'][key][field], step, dec)
                s.changed.connect(lambda val, k=key, f=field: self.set_inter(k, f, val))
                gl.addWidget(s)
                self.inter_widgets[(key, field)] = s
            v.addWidget(g)
        g = W.QGroupBox('Contact (capture)')
        gl = W.QVBoxLayout(g)
        for key, label in (('cap_ab', 'Quand un CRS touche un manifestant'), ('cap_ba', 'Quand un manifestant touche un CRS')):
            row = W.QHBoxLayout()
            row.addWidget(W.QLabel(label))
            cb = W.QComboBox()
            cb.addItems(S.CAPTURE)
            cb.setCurrentText(self.P[key])
            cb.currentTextChanged.connect(lambda val, k=key: self.set_param(k, val))
            row.addWidget(cb, 1)
            gl.addLayout(row)
            self.widgets[key] = cb
        self._slider(gl, 'cap_radius', 'Distance de contact (% haut.)', 0.5, 15, 0.5, 1)
        self._slider(gl, 'cap_hold', 'Duree (s)', 0.2, 10, 0.2, 1)
        v.addWidget(g)
        v.addStretch(1)
        return w

    def _tab_animation(self):
        w, v = self._tab()
        self._combo(v, 'anim_mode', 'Animation des boucles', S.ANIM_MODES)
        self._slider(v, 'anim_speed', 'Vitesse de lecture', 0.1, 4, 0.05, 2, 'x')
        self._slider(v, 'cadence', 'Cadence saccadee (images/s)', 1, 30, 1, 0, ' i/s')
        self._check(v, 'sync_speed', 'Lecture liee a la vitesse de deplacement (pas glisses)')
        self._check(v, 'pos_stepped', 'Deplacements saccades (stop motion sur les positions)')
        self._slider(v, 'boil', 'Tremblement des positions (px)', 0, 12, 0.5, 1)
        self._check(v, 'flip_dir', 'Retourner selon le sens de marche (sinon au hasard)')
        v.addWidget(W.QLabel('Fluide : lecture a la cadence native de la boucle.\nSaccade : une image sur N, comme une animation image par image.\n'
                             'Ping-pong : aller-retour (pas de saut au bouclage).\nAleatoires : image tiree au hasard a chaque pas.\n'
                             'Pose figee : une pose par personnage, leger balancement.'))
        v.addStretch(1)
        return w

    def _tab_canvas(self):
        w, v = self._tab()
        cb = self._combo(v, 'canvas', 'Format du canvas', CANVAS_PRESETS)
        cb.currentTextChanged.disconnect()
        cb.currentTextChanged.connect(self.apply_canvas)
        self._slider(v, 'cw', 'Largeur (px)', 320, 7680, 2).changed.connect(self.canvas_manual)
        self._slider(v, 'ch', 'Hauteur (px)', 240, 4320, 2).changed.connect(self.canvas_manual)
        v.addWidget(W.QLabel('Le canvas suit la resolution de rendu (onglet Export) en gardant son format ;\n'
                             'modifier la largeur / hauteur ici passe la resolution sur "Taille du canvas".'))
        row = W.QHBoxLayout()
        row.addWidget(W.QLabel('Couleur de fond'))
        self.bg_btn = W.QPushButton(self.P['bg'])
        self.bg_btn.clicked.connect(self.pick_bg)
        row.addWidget(self.bg_btn, 1)
        v.addLayout(row)
        row = W.QHBoxLayout()
        self.bgimg_edit = W.QLineEdit(self.P['bg_image'])
        self.bgimg_edit.setPlaceholderText('Image de fond (optionnel)')
        self.bgimg_edit.editingFinished.connect(lambda: self.set_param('bg_image', self.bgimg_edit.text()))
        bb = W.QPushButton('Image...')
        bb.clicked.connect(self.pick_bgimg)
        row.addWidget(self.bgimg_edit, 1)
        row.addWidget(bb)
        v.addLayout(row)
        self._check(v, 'transparent', 'Fond transparent a l\'export (formats avec alpha)')
        self._slider(v, 'size_pct', 'Taille des personnages (% haut.)', 3, 80, 0.5, 1)
        self._slider(v, 'smin', 'Echelle au fond (haut)', 0.1, 2, 0.05, 2)
        self._slider(v, 'smax', 'Echelle au premier plan (bas)', 0.1, 3, 0.05, 2)
        self._slider(v, 'gamma', 'Courbe de perspective', 0.3, 3, 0.05, 2)
        self._slider(v, 'shadow', 'Ombres au sol', 0, 1, 0.05, 2)
        v.addStretch(1)
        return w

    def _tab_export(self):
        w, v = self._tab()
        self._combo(v, 'fmt', 'Format', R.FORMATS)
        self._combo(v, 'res', 'Resolution (le canvas s\'y adapte)', R.RESOLUTIONS).currentTextChanged.connect(self.sync_canvas_to_res)
        self._slider(v, 'fps', 'Images / seconde', 12, 60, 1)
        self._slider(v, 'seconds', 'Duree (s)', 1, 120, 0.5, 1, ' s')
        self._slider(v, 'warmup', 'Pre-roulement (s)', 0, 20, 0.5, 1, ' s')
        row = W.QHBoxLayout()
        self.out_edit = W.QLineEdit(self.P['out_dir'])
        self.out_edit.editingFinished.connect(lambda: self.set_param('out_dir', self.out_edit.text()))
        bb = W.QPushButton('Dossier...')
        bb.clicked.connect(self.pick_out)
        row.addWidget(W.QLabel('Sortie'))
        row.addWidget(self.out_edit, 1)
        row.addWidget(bb)
        v.addLayout(row)
        row = W.QHBoxLayout()
        self.name_edit = W.QLineEdit(self.P['out_name'])
        self.name_edit.editingFinished.connect(lambda: self.set_param('out_name', self.name_edit.text()))
        row.addWidget(W.QLabel('Nom'))
        row.addWidget(self.name_edit, 1)
        v.addLayout(row)
        self.exp_btn = W.QPushButton('Exporter')
        self.exp_btn.clicked.connect(self.do_export)
        self.exp_bar = W.QProgressBar()
        self.exp_msg = W.QLabel('')
        self.exp_msg.setWordWrap(True)
        v.addWidget(self.exp_btn)
        v.addWidget(self.exp_bar)
        v.addWidget(self.exp_msg)
        v.addWidget(W.QLabel('Les formats "transparent" gardent l\'alpha (WebM VP9, ProRes 4444, PNG)\net ignorent la couleur de fond.\n'
                             'Resolutions superieures aux sources : les personnages sont agrandis (plus doux).'))
        v.addStretch(1)
        return w

    # ---- equipe editee
    def select_team(self, i):
        self.cur_team = i
        self.bind_team()

    def bind_team(self):
        T = self.P['teams'][self.cur_team]
        for k, w in self.team_widgets.items():
            if isinstance(w, Slider):
                w.set(T[k])
            else:
                w.blockSignals(True)
                w.setCurrentText(T[k])
                w.blockSignals(False)

    def set_team_param(self, k, v):
        self.P['teams'][self.cur_team][k] = v
        if k in self.TEAM_REGEN:
            self.regen()

    def set_inter(self, key, field, v):
        self.P['inter'][key][field] = v

    def calm_team(self):
        T = self.P['teams'][self.cur_team]
        T.update(home=max(T.get('home', 0), 2.0), home_r=min(T.get('home_r', 5), 4.0), damp=max(T.get('damp', 1.2), 3.0),
                 smooth=max(T.get('smooth', 0), 0.5), force=min(T.get('force', 6), 3.0), wander=T['wander'] * 0.5)
        self.bind_team()

    def anchor_now(self):
        if self.crowd is not None:
            self.crowd.anchor_here()

    def apply_move_preset(self, name):
        for k, v in S.MOVE_PRESETS[name].items():
            self.P['teams'][self.cur_team][k] = v
        self.bind_team()

    def apply_scenario(self, name):
        self.P['scenario'] = name
        S.apply_scenario(self.P, name)
        self.bind_team()
        for (key, field), w in self.inter_widgets.items():
            val = self.P['inter'][key][field]
            if isinstance(w, Slider):
                w.set(val)
            else:
                w.blockSignals(True)
                w.setCurrentText(val)
                w.blockSignals(False)
        for k in ('cap_ab', 'cap_ba'):
            self.widgets[k].blockSignals(True)
            self.widgets[k].setCurrentText(self.P[k])
            self.widgets[k].blockSignals(False)
        self.regen()

    # ---- bibliotheque
    def pick_folder(self, key, edit):
        d = W.QFileDialog.getExistingDirectory(self, 'Dossier des sequences', edit.text())
        if d:
            edit.setText(d)
            self.change_folder(key, d)

    def change_folder(self, key, path):
        if path != self.P[key]:
            self.P[key] = path
            self.set_folder()

    def set_folder(self):
        if self.loader:
            self.loader.stop = True
            self.loader.wait()
        seqs = []
        for key, team, pre in (('folder_crs', 0, 'crs:'), ('folder_manif', 1, 'manif:')):
            if os.path.isdir(self.P[key]):
                seqs += L.scan(self.P[key], team, pre)
        for s in seqs:
            if s.id in self.P['team_override']:
                s.team = self.P['team_override'][s.id]
        self.seqs = seqs
        self.by_id = {s.id: s for s in self.seqs}
        self.rend = R.Renderer(self.by_id)
        if self.saved_enabled is not None:
            self.enabled = {i for i in self.saved_enabled if i in self.by_id or '#' in i}
        else:
            self.enabled = self.default_selection()
        self.saved_enabled = None
        if not any(i in self.by_id for i in self.enabled):
            self.enabled = self.default_selection() | {i for i in self.enabled if '#' in i}
        self.clip_combo.blockSignals(True)
        self.clip_combo.clear()
        self.clip_combo.addItem('Tous les clips')
        for c in sorted({s.clip for s in self.seqs}):
            self.clip_combo.addItem(c)
        self.clip_combo.blockSignals(False)
        self.refill_list()
        self.loader = None
        self.sync_loading()
        self.regen()

    DEFAULT_PER_TEAM = 80

    def default_selection(self):
        """Au plus DEFAULT_PER_TEAM boucles par equipe, reparties entre les clips (les plus longues d'abord)."""
        out = set()
        for t in (0, 1):
            by_clip = {}
            for s in self.seqs:
                if s.team == t and s.n >= self.P['min_frames']:
                    by_clip.setdefault(s.clip, []).append(s)
            for v in by_clip.values():
                v.sort(key=lambda s: -s.n)
            lists = sorted(by_clip.values(), key=lambda v: v[0].id)
            k = 0
            picked = 0
            while picked < self.DEFAULT_PER_TEAM and any(lists):
                v = lists[k % len(lists)]
                if v:
                    out.add(v.pop(0).id)
                    picked += 1
                k += 1
        return out

    def schedule_sync(self):
        if not hasattr(self, '_sync_timer'):
            self._sync_timer = QtCore.QTimer(self)
            self._sync_timer.setSingleShot(True)
            self._sync_timer.timeout.connect(self.sync_loading)
        self._sync_timer.start(500)

    def sync_loading(self):
        """Charge en memoire les boucles cochees (et libere les autres)."""
        if self.loader and self.loader.isRunning():
            self._resync = True
            return
        busy = self.exporter is not None and self.exporter.isRunning()
        if not busy:
            for s in self.seqs:
                if s.id not in self.enabled and s.virtual is None and s.frames is not None and s is not self.current_seq():
                    s.frames = None
        todo = [s for s in self.seqs if s.id in self.enabled and s.frames is None and s.virtual is None]
        todo.sort(key=lambda s: s.id)
        if not todo:
            self.rebuild_virtual()
            return
        self._resync = False
        self.loader = Loader(todo)
        self.loader.progress.connect(self.on_load_progress)
        self.loader.finished.connect(self.on_loader_done)
        self.loader.start()

    def on_loader_done(self):
        if self.loader and self.loader.skipped:
            self.lib_info.setText(f'Memoire limite : {self.loader.skipped} boucles cochees non chargees (decochez-en).')
        self.rebuild_virtual()
        if getattr(self, '_resync', False):
            self.sync_loading()

    def rebuild_virtual(self):
        """recree les boucles virtuelles memorisees (apres chargement des sources)."""
        if self.loader and self.loader.stop:
            return
        added = False
        for d in self.P['loops']:
            src = self.by_id.get(d['src'])
            if src is None:
                continue
            s = L.make_loop(src, d['start'], d['n'], d['mode'], d.get('xf', 3), d.get('team'))
            if s.id not in self.by_id:
                self.add_seq(s, enable=s.id in self.enabled or True)
                added = True
        if added:
            self.refill_list()

    def add_seq(self, s, enable=True):
        self.seqs.append(s)
        self.by_id[s.id] = s
        self.rend.seqs = self.by_id
        if enable:
            self.enabled.add(s.id)

    def on_load_progress(self, i, n):
        self.lib_info.setText(f'Chargement des boucles : {i}/{n}' if i < n else f'{n} boucles chargees')

    def visible_seqs(self):
        clip = self.clip_combo.currentText()
        m = self.min_spin.value()
        tf = self.team_filter.currentIndex() - 1
        return [s for s in self.seqs if s.n >= m and (clip == 'Tous les clips' or s.clip == clip) and (tf < 0 or s.team == tf)]

    def refill_list(self, *_):
        self.P['min_frames'] = self.min_spin.value()
        self.list.blockSignals(True)
        self.list.clear()
        self._items = {}
        self._thumb_queue = []
        for s in self.visible_seqs():
            tag = TEAM_TAG[s.team] + (' boucle' if s.virtual else '')
            if L.ANON:      # captures publiques : pas de noms de clips / d'auteurs
                it = W.QListWidgetItem(f'[{tag}] clip {abs(hash(s.clip)) % 90 + 10}\nsequence {abs(hash(s.id)) % 900 + 100}  -  {s.n} img')
            else:
                it = W.QListWidgetItem(f'[{tag}] {s.clip[:24]}\n{s.name}  -  {s.n} img')
            self._items[s.id] = it
            if s.frames is not None or s.thumb is not None:
                self.set_icon(it, s)
            else:
                self._thumb_queue.append(s.id)
            it.setBackground(TEAM_BG[s.team])
            it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.Checked if s.id in self.enabled else QtCore.Qt.Unchecked)
            it.setData(QtCore.Qt.UserRole, s.id)
            self.list.addItem(it)
        self.list.blockSignals(False)
        self.update_info()

    def set_icon(self, it, s):
        try:
            if s.frames is not None:
                pm = seq_to_pixmap(s.frames, s.n // 2, 64)
            else:
                t = L.thumbnail(s)
                qi = QtGui.QImage(t.data, t.shape[1], t.shape[0], t.shape[1] * 3, QtGui.QImage.Format_RGB888).copy()
                pm = QtGui.QPixmap.fromImage(qi)
            it.setIcon(QtGui.QIcon(pm))
        except Exception:
            pass

    def fill_thumbs(self):
        """remplit les vignettes par petits lots pour ne pas bloquer l'interface."""
        t0 = time.perf_counter()
        q = self._thumb_queue
        while q and time.perf_counter() - t0 < 0.03:
            sid = q.pop(0)
            it, s = self._items.get(sid), self.by_id.get(sid)
            if it is not None and s is not None:
                self.set_icon(it, s)

    def update_info(self):
        c = [sum(1 for i in self.enabled if i in self.by_id and self.by_id[i].team == t) for t in (0, 1)]
        self.lib_info.setText(f'Boucles actives : {c[0]} CRS, {c[1]} manifestants  (sur {len(self.seqs)})')

    def on_item_changed(self, it):
        sid = it.data(QtCore.Qt.UserRole)
        (self.enabled.add if it.checkState() == QtCore.Qt.Checked else self.enabled.discard)(sid)
        self.update_info()
        self.schedule_sync()

    def _set_visible_checked(self, fn):
        self.list.blockSignals(True)
        for i in range(self.list.count()):
            it = self.list.item(i)
            on = fn(it)
            it.setCheckState(QtCore.Qt.Checked if on else QtCore.Qt.Unchecked)
            (self.enabled.add if on else self.enabled.discard)(it.data(QtCore.Qt.UserRole))
        self.list.blockSignals(False)
        self.update_info()
        self.schedule_sync()

    def check_visible(self, on):
        self._set_visible_checked(lambda it: on)

    def invert_visible(self):
        self._set_visible_checked(lambda it: it.checkState() != QtCore.Qt.Checked)

    def current_seq(self):
        cur = self.list.currentItem()
        return None if cur is None else self.by_id.get(cur.data(QtCore.Qt.UserRole))

    def swap_team(self):
        s = self.current_seq()
        if s is None:
            return
        s.team = 1 - s.team
        self.P['team_override'][s.id] = s.team
        if s.virtual:
            s.virtual['team'] = s.team
            for d in self.P['loops']:
                if self._loop_id(d) == s.id:
                    d['team'] = s.team
        self.refill_list()

    def _loop_id(self, d):
        src = self.by_id.get(d['src'])
        if src is None:
            return None
        n = max(2, min(d['n'], src.n - max(0, min(d['start'], src.n - 1))))
        return f"{src.id}#{d['start'] + 1}+{n}:{d['mode'][:4]}" + (f"x{d.get('xf', 3)}" if d['mode'] == 'Fondu enchaine' else '')

    def on_select(self, cur, _prev):
        self.mini_i = 0.0
        s = self.current_seq()
        if s is not None:
            self.flip_chk.blockSignals(True)
            self.flip_chk.setChecked(s.id in self.P['flip_ids'])
            self.flip_chk.blockSignals(False)

    def on_flip_toggle(self, on):
        s = self.current_seq()
        if s is None:
            return
        f = set(self.P['flip_ids'])
        (f.add if on else f.discard)(s.id)
        self.P['flip_ids'] = sorted(f)

    def create_loop(self):
        s = self.current_seq()
        if s is None or s.virtual:
            self.lib_info.setText('Selectionnez d\'abord une sequence source (pas une boucle deja creee).')
            return
        dlg = LoopDialog(self, s)
        while dlg.exec() == W.QDialog.Accepted and dlg.result is not None:
            ns = dlg.result
            if ns.id not in self.by_id:
                self.add_seq(ns)
                self.P['loops'].append(dict(ns.virtual))
            dlg.result = None
        self.refill_list()

    def delete_loop(self):
        s = self.current_seq()
        if s is None or not s.virtual:
            return
        self.seqs = [x for x in self.seqs if x.id != s.id]
        self.by_id.pop(s.id, None)
        self.enabled.discard(s.id)
        self.P['loops'] = [d for d in self.P['loops'] if self._loop_id(d) != s.id]
        self.refill_list()

    def auto_slice(self):
        n = self.auto_n.value()
        made = 0
        for s in list(self.seqs):
            if s.virtual or s.id not in self.enabled or s.n < n:
                continue
            for st in range(0, s.n - n + 1, n)[:6]:
                ns = L.make_loop(s, st, n, 'Coupe nette')
                if ns.id not in self.by_id:
                    self.add_seq(ns)
                    self.P['loops'].append(dict(ns.virtual))
                    made += 1
            self.enabled.discard(s.id)      # la source est remplacee par ses morceaux
        self.refill_list()
        self.lib_info.setText(f'{made} boucles de {n} images creees.')

    def mini_tick(self):
        s = self.current_seq()
        if s is None:
            return
        if s.frames is None:
            L.load(s)
        self.mini_i += s.fps * 0.04
        self.mini.setPixmap(seq_to_pixmap(s.frames, int(self.mini_i), 180, s.id in self.P['flip_ids']))

    # ---- parametres globaux
    def set_param(self, k, v):
        self.P[k] = v
        if k in self.REGEN:
            self.regen()

    def apply_canvas(self, name):
        self.P['canvas'] = name
        wh = CANVAS_PRESETS.get(name)
        if wh:
            self.P['cw'], self.P['ch'] = wh
            self.widgets['cw'].set(wh[0])
            self.widgets['ch'].set(wh[1])
            if not self.sync_canvas_to_res():
                self.regen()

    def sync_canvas_to_res(self, *_):
        """le canvas de simulation prend la largeur de la resolution de rendu choisie (format conserve)."""
        wout = R.RESOLUTIONS.get(self.P['res'], 0)
        if not wout or wout == self.P['cw']:
            return False
        ratio = self.P['ch'] / self.P['cw']
        self.P['cw'] = int(wout)
        self.P['ch'] = int(round(wout * ratio / 2)) * 2
        self.widgets['cw'].set(self.P['cw'])
        self.widgets['ch'].set(self.P['ch'])
        self.regen()
        return True

    def canvas_manual(self, *_):
        """largeur / hauteur modifiees a la main : le rendu reprend la taille du canvas."""
        cb = self.widgets['res']
        if cb.currentText() != 'Taille du canvas':
            cb.blockSignals(True)
            cb.setCurrentText('Taille du canvas')
            cb.blockSignals(False)
            self.P['res'] = 'Taille du canvas'

    def pick_bg(self):
        c = W.QColorDialog.getColor(QtGui.QColor(self.P['bg']), self)
        if c.isValid():
            self.P['bg'] = c.name()
            self.bg_btn.setText(c.name())

    def pick_bgimg(self):
        f, _ = W.QFileDialog.getOpenFileName(self, 'Image de fond', '', 'Images (*.png *.jpg *.jpeg *.webp)')
        if f:
            self.bgimg_edit.setText(f)
            self.set_param('bg_image', f)

    def pick_out(self):
        d = W.QFileDialog.getExistingDirectory(self, 'Dossier de sortie', self.out_edit.text())
        if d:
            self.out_edit.setText(d)
            self.P['out_dir'] = d

    # ---- foule
    def pools(self):
        return [sorted(i for i in self.enabled if i in self.by_id and self.by_id[i].team == t) for t in (0, 1)]

    def make_crowd(self, P=None, pools=None):
        P = P or self.P
        return S.Crowd(P['cw'], P['ch'], P, pools or self.pools(), P['seed'])

    def regen(self):
        self.crowd = self.make_crowd()

    def new_seed(self):
        self.P['seed'] = int(np.random.randint(0, 999))
        self.widgets['seed'].set(self.P['seed'])
        self.regen()

    def clear_goal(self):
        if self.crowd is not None:
            self.crowd.goal_override = [None, None]

    def on_preview_click(self, fx, fy):
        if self.crowd is not None:
            self.crowd.goal_override[self.cur_team] = np.array([fx * self.crowd.W, fy * self.crowd.H])
            T = self.P['teams'][self.cur_team]
            if T['goal'] == 0:
                T['goal'] = 1.5
                self.bind_team()

    def toggle_play(self):
        self.playing = not self.playing
        self.play_btn.setText('Pause' if self.playing else 'Lecture')

    def tick(self):
        if not self.playing or self.crowd is None:
            return
        t0 = time.perf_counter()
        pools = self.pools()
        if not (pools[0] or pools[1]):
            self.status.setText('Aucune boucle active : cochez des boucles dans la bibliotheque.')
            return
        if self.crowd.seq_ids != pools[0] + pools[1]:
            self.crowd.assign(pools)
        self.crowd.step(1 / 30, self.P)
        pw = max(320, min(self.preview.width(), 960))
        img = self.rend.render(self.crowd, self.P, pw / self.crowd.W)
        self.preview.show_image(img)
        dt = time.perf_counter() - t0
        self.fps_disp = 0.9 * self.fps_disp + 0.1 / max(dt, 1e-3)
        c = [int(((self.crowd.team == t) & self.crowd.alive).sum()) for t in (0, 1)]
        miss = ' (equipe sans boucle : invisible)' if (not pools[0] and c[0]) or (not pools[1] and c[1]) else ''
        self.status.setText(f'{c[0]} CRS, {c[1]} manifestants, apercu {img.shape[1]}x{img.shape[0]}, ~{self.fps_disp:.0f} i/s{miss}')

    # ---- export
    def do_export(self):
        if self.exporter and self.exporter.isRunning():
            self.exporter.cancel = True
            return
        pools = self.pools()
        if not (pools[0] or pools[1]):
            self.exp_msg.setText('Aucune boucle active.')
            return
        for i in pools[0] + pools[1]:
            L.load(self.by_id[i])
        os.makedirs(self.P['out_dir'], exist_ok=True)
        ext = R.FORMATS[self.P['fmt']]['ext']
        path = os.path.join(self.P['out_dir'], f"{self.P['out_name']}_{time.strftime('%H%M%S')}.{ext}")
        P = copy.deepcopy(self.P)
        rend = R.Renderer(self.by_id)
        self.exporter = Exporter(lambda: S.Crowd(P['cw'], P['ch'], P, pools, P['seed']), rend, P, path, P['fmt'],
                                 R.RESOLUTIONS[P['res']], P['fps'], P['seconds'])
        self.exporter.progress.connect(lambda i, n: (self.exp_bar.setMaximum(n), self.exp_bar.setValue(i)))
        self.exporter.done.connect(self.on_exp_done)
        self.exporter.failed.connect(self.on_exp_fail)
        self.exp_btn.setText('Annuler l\'export')
        self.exp_msg.setText('Rendu en cours...')
        self.exporter.start()

    def on_exp_done(self, out):
        self.exp_btn.setText('Exporter')
        self.exp_msg.setText('Termine : ' + out)

    def on_exp_fail(self, msg):
        self.exp_btn.setText('Exporter')
        self.exp_msg.setText('Echec : ' + msg)

    # ---- projets
    def save_project(self):
        f, _ = W.QFileDialog.getSaveFileName(self, 'Enregistrer le projet', os.path.join(HERE, 'projet.json'), 'JSON (*.json)')
        if f:
            self.save_settings(f)

    def open_project(self):
        f, _ = W.QFileDialog.getOpenFileName(self, 'Ouvrir un projet', HERE, 'JSON (*.json)')
        if not f:
            return
        self.load_settings(f)
        for k, w in self.widgets.items():
            if isinstance(w, Slider):
                w.set(self.P[k])
            elif isinstance(w, W.QComboBox):
                w.setCurrentText(self.P[k])
            elif isinstance(w, W.QCheckBox):
                w.setChecked(self.P[k])
        self.bg_btn.setText(self.P['bg'])
        self.bind_team()
        for (key, field), w in self.inter_widgets.items():
            val = self.P['inter'][key][field]
            w.set(val) if isinstance(w, Slider) else w.setCurrentText(val)
        self.edit_folder_crs.setText(self.P['folder_crs'])
        self.edit_folder_manif.setText(self.P['folder_manif'])
        self.set_folder()


def main():
    app = W.QApplication(sys.argv)
    app.setStyle('Fusion')
    m = Main()
    m.show()
    sys.exit(app.exec())
