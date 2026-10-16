"""Moteur de foule : placements, boids (nuees de points), indices d'animation des boucles.
Toutes les positions sont en pixels du canvas de projet (origine en haut a gauche, y = pieds a l'ecran)."""
import math

import numpy as np

PLACEMENTS = ['Foule dense (nuage)', 'Disperse (aleatoire)', 'Grille / rangs', 'Lignes (cordon)',
              'Cercle / anneau', 'Deux lignes face a face', 'Colonne (cortege)', 'V (formation)',
              'Mur dense en bas de cadre', 'Disque plein']

# preset de mouvement -> valeurs des curseurs
# forces de stabilite (par equipe) : ancrage au point d'origine, frottement, lissage, force de pilotage maximale
MOVE_EXTRA = dict(home=0.0, home_r=5.0, damp=1.2, smooth=0.0, force=6.0)

MOVE_PRESETS = {
    'Ancre (garde sa place)': dict(speed=5, sep=1.2, ali=0.4, coh=0.3, radius=10, sep_radius=4, goal=0.0, swirl=0.0,
                                   wander=0.3, wind_x=0.0, wind_y=0.0, depth_mob=0.5, boundary='Rappel doux',
                                   home=2.5, home_r=3.0, damp=3.0, smooth=0.5, force=2.5),
    'Nuee (boids classique)': dict(speed=9, sep=1.6, ali=1.0, coh=0.8, radius=14, sep_radius=4.5, goal=0.0, swirl=0.0,
                                   wander=0.6, wind_x=0.0, wind_y=0.0, depth_mob=0.6, boundary='Rappel doux'),
    'Marche en flux (cortege)': dict(speed=7, sep=1.4, ali=2.5, coh=0.2, radius=12, sep_radius=4, goal=0.0, swirl=0.0,
                                     wander=0.25, wind_x=1.0, wind_y=0.0, depth_mob=0.25, boundary='Reapparait de l autre cote'),
    'Charge vers un point': dict(speed=16, sep=1.2, ali=1.2, coh=0.1, radius=12, sep_radius=3.5, goal=2.4, swirl=0.0,
                                 wander=0.3, wind_x=0.0, wind_y=0.0, depth_mob=0.7, boundary='Libre'),
    'Moulin (rotation)': dict(speed=8, sep=1.4, ali=0.8, coh=0.5, radius=14, sep_radius=4, goal=0.0, swirl=2.2,
                              wander=0.3, wind_x=0.0, wind_y=0.0, depth_mob=0.8, boundary='Rappel doux'),
    'Foule sur place (agitee)': dict(speed=2.5, sep=1.8, ali=0.2, coh=0.6, radius=10, sep_radius=4.5, goal=0.0,
                                     swirl=0.0, wander=1.4, wind_x=0.0, wind_y=0.0, depth_mob=0.5, boundary='Rappel doux'),
    'Fuite (panique)': dict(speed=20, sep=1.0, ali=0.6, coh=0.0, radius=12, sep_radius=3.5, goal=-3.0, swirl=0.0,
                            wander=0.8, wind_x=0.0, wind_y=0.0, depth_mob=0.8, boundary='Rebond'),
    'Affrontement (deux camps)': dict(speed=11, sep=1.4, ali=1.0, coh=0.3, radius=12, sep_radius=4, goal=1.6, swirl=0.0,
                                      wander=0.4, wind_x=0.0, wind_y=0.0, depth_mob=0.4, boundary='Rebond'),
}

for _p in MOVE_PRESETS.values():
    for _k, _v in MOVE_EXTRA.items():
        _p.setdefault(_k, _v)

BOUNDARIES = ['Rappel doux', 'Rebond', 'Reapparait de l autre cote', 'Libre']
ANIM_MODES = ['Fluide (cadence native)', 'Saccade (stop motion)', 'Ping-pong', 'Images aleatoires (tremblement)',
              'Pose figee (balancement)']


def placement(preset, n, W, H, rng, cx, cy, zw, zh, rows, direction_deg):
    """-> pos (n,2), group (n,), goals (n,2) (point cible par individu)"""
    cx, cy, zw, zh = cx * W, cy * H, zw * W, zh * H
    group = np.zeros(n, int)
    goals = np.tile([cx, cy], (n, 1)).astype(float)
    u = rng.random((n, 2))
    if preset == PLACEMENTS[0]:
        pos = np.c_[cx + rng.normal(0, zw / 5, n), cy + rng.normal(0, zh / 5, n)]
    elif preset == PLACEMENTS[1]:
        pos = np.c_[cx + (u[:, 0] - .5) * zw, cy + (u[:, 1] - .5) * zh]
    elif preset == PLACEMENTS[2]:
        r = max(1, int(round(math.sqrt(n * zh / max(zw, 1)))))
        c = int(math.ceil(n / r))
        i = np.arange(n)
        pos = np.c_[cx + ((i % c) / max(c - 1, 1) - .5) * zw, cy + ((i // c) / max(r - 1, 1) - .5) * zh]
        pos += rng.normal(0, 0.012 * zw, (n, 2))
    elif preset == PLACEMENTS[3]:
        r = max(1, rows)
        row = np.arange(n) % r
        pos = np.c_[cx + (rng.random(n) - .5) * zw, cy + (row / max(r - 1, 1) - .5) * zh]
        pos[:, 0] += rng.normal(0, 0.006 * zw, n)
    elif preset == PLACEMENTS[4]:
        a = rng.random(n) * 2 * np.pi
        rr = 1 - 0.12 * rng.random(n)
        pos = np.c_[cx + np.cos(a) * zw / 2 * rr, cy + np.sin(a) * zh / 2 * rr]
    elif preset == PLACEMENTS[5]:
        group = (np.arange(n) % 2).astype(int)
        side = np.where(group == 0, -1, 1)
        pos = np.c_[cx + side * zw / 2 + rng.normal(0, 0.02 * zw, n), cy + (u[:, 1] - .5) * zh]
        goals = np.c_[cx - side * zw / 2, cy + (u[:, 1] - .5) * zh]
    elif preset == PLACEMENTS[6]:
        r = max(1, rows)
        pos = np.c_[cx + (u[:, 0] - .5) * zw, cy + ((np.arange(n) % r) / max(r - 1, 1) - .5) * zh * 0.5]
        pos[:, 1] += rng.normal(0, 0.01 * H, n)
    elif preset == PLACEMENTS[7]:
        arm = np.where(rng.random(n) < .5, -1, 1)
        t = rng.random(n)
        pos = np.c_[cx + arm * t * zw / 2, cy + t * zh / 2 - zh / 2] + rng.normal(0, 0.01 * H, (n, 2))
    elif preset == PLACEMENTS[8]:
        pos = np.c_[(rng.random(n)) * W, H * (0.97 - 0.30 * rng.random(n) ** 1.5)]
    else:  # disque
        a = rng.random(n) * 2 * np.pi
        rr = np.sqrt(rng.random(n))
        pos = np.c_[cx + np.cos(a) * rr * zw / 2, cy + np.sin(a) * rr * zh / 2]
    ang = math.radians(direction_deg)
    return pos, group, goals, np.array([math.cos(ang), math.sin(ang)])


TEAM_NAMES = ['CRS', 'Manifestants']
PAIR_MODES = ['Ignorer', 'Poursuivre le plus proche', 'Poursuivre la cible unique (leader)', 'Encercler',
              'Fuir le plus proche', 'Eviter', 'Escorter / suivre', 'Barrer la route']
CAPTURE = ['Rien', 'Ralentir', 'Figer', 'Retirer', 'Convertir (change de camp)']
TEAM_KEYS = ['n', 'place', 'zx', 'zy', 'zw', 'zh', 'rows', 'direction', 'speed', 'sep', 'ali', 'coh', 'radius', 'sep_radius',
             'goal', 'swirl', 'wander', 'wind_x', 'wind_y', 'depth_mob', 'boundary', 'size_mult', 'anim_mult', 'gx', 'gy',
             'home', 'home_r', 'damp', 'smooth', 'force']


def team_defaults(i):
    d = dict(n=60 if i == 0 else 40, place=PLACEMENTS[0], zx=0.35 if i == 0 else 0.65, zy=0.68, zw=0.45, zh=0.4, rows=3,
             direction=0 if i == 0 else 180, size_mult=1.0, anim_mult=1.0, gx=-1.0, gy=0.68)
    d.update(MOVE_PRESETS['Ancre (garde sa place)'])
    return d


def inter_defaults():
    return dict(mode=PAIR_MODES[0], strength=1.5, range=60.0, lead=0.5)


def _unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-6)


class Crowd:
    """Deux equipes (CRS / manifestants) qui interagissent.
    P['teams'] = [dict, dict] ; P['inter'] = {'ab': CRS->manif, 'ba': manif->CRS}.
    pools : [ids_equipe0, ids_equipe1] (identifiants des boucles actives de chaque equipe)."""

    def __init__(self, W, H, P, pools, seed):
        self.W, self.H = W, H
        self.rng = np.random.default_rng(seed)
        r = self.rng
        pos, team, goals, vel0 = [], [], [], []
        for t, T in enumerate(P['teams']):
            n = int(T['n'])
            if n <= 0:
                continue
            p, g, gl, d = placement(T['place'], n, W, H, r, T['zx'], T['zy'], T['zw'], T['zh'], T['rows'], T['direction'])
            pos.append(p)
            team.append(np.full(n, t))
            if T.get('gx', -1) >= 0:
                gl = np.tile([T['gx'] * W, T['gy'] * H], (n, 1))
            goals.append(gl)
            vel0.append(np.tile(d, (n, 1)) * (T['speed'] / 100 * H) * (0.4 + 0.6 * r.random((n, 1))))
        if pos:
            self.pos = np.vstack(pos).astype(float)
            self.team = np.concatenate(team)
            self.goals = np.vstack(goals)
            self.vel = np.vstack(vel0)
        else:
            self.pos = np.zeros((0, 2))
            self.team = np.zeros(0, int)
            self.goals = np.zeros((0, 2))
            self.vel = np.zeros((0, 2))
        n = len(self.pos)
        self.n = n
        self.wander_ang = r.random(n) * 2 * np.pi
        self.spd_rand = 0.8 + 0.4 * r.random(n)
        self.scale_rand = 0.92 + 0.16 * r.random(n)
        self.size_mult = np.ones(n)
        self.phase = r.random(n) * 10.0
        self.rate_rand = 0.85 + 0.3 * r.random(n)
        self.flip_rand = r.random(n) < .5
        self.alive = np.ones(n, bool)
        self.frozen_until = np.zeros(n)
        self.slow = np.ones(n)
        self.seqs = np.zeros(n, int)
        self.seq_ids = []
        self.pool_idx = [[], []]
        self.leader = [int(np.argmax(self.team == t)) if (self.team == t).any() else -1 for t in range(2)]
        self.assign(pools)
        self.t = 0.0
        self.disp = self.pos.copy()
        self.disp_step = -1
        self.goal_override = [None, None]
        self._vmax = np.ones(n)
        self.home = self.pos.copy()          # point d'ancrage de chaque personnage
        self.acc_prev = np.zeros((n, 2))

    def anchor_here(self):
        """les positions actuelles deviennent les points d'ancrage."""
        self.home = self.pos.copy()

    def assign(self, pools):
        """(re)attribue une boucle a chaque personnage parmi celles de son equipe ; equipe sans boucle = invisible."""
        self.seq_ids = list(pools[0]) + list(pools[1])
        self.pool_idx = [list(range(len(pools[0]))), list(range(len(pools[0]), len(self.seq_ids)))]
        for t in range(2):
            m = self.team == t
            if self.pool_idx[t] and m.any():
                self.seqs[m] = self.rng.choice(self.pool_idx[t], int(m.sum()))
        self.visible = np.array([bool(self.pool_idx[t]) for t in self.team], bool) if self.n else np.zeros(0, bool)

    def _arr(self, P, key):
        v = np.array([float(P['teams'][t].get(key, MOVE_EXTRA.get(key, 0.0))) for t in range(2)])
        return v[self.team]

    def step(self, dt, P):
        n = self.n
        if n == 0:
            return
        H, W = self.H, self.W
        pos, vel, team = self.pos, self.vel, self.team
        al = self.alive
        vmax = self._arr(P, 'speed') / 100 * H * self.spd_rand * self.slow
        self._vmax = vmax
        rad = self._arr(P, 'radius') / 100 * H
        srad = self._arr(P, 'sep_radius') / 100 * H
        frozen = self.frozen_until > self.t
        diff = pos[:, None, :] - pos[None, :, :]
        dist = np.sqrt((diff ** 2).sum(-1)) + 1e-6
        dist[:, ~al] = 1e9
        dist[~al, :] = 1e9
        np.fill_diagonal(dist, 1e9)
        same = team[:, None] == team[None, :]
        nb = (dist < rad[:, None]) & same
        cnt = nb.sum(1, keepdims=True)
        acc = np.zeros((n, 2))
        sm = dist < srad[:, None]
        sv = ((diff / dist[..., None]) * (srad[:, None] / dist)[..., None] * sm[..., None]).sum(1)
        sn = np.linalg.norm(sv, axis=1, keepdims=True) + 1e-6
        acc += self._arr(P, 'sep')[:, None] * sv / sn * np.minimum(sn, 3.0) * vmax[:, None] * 1.5
        has = cnt[:, 0] > 0
        if has.any():
            mv = (nb[..., None] * vel[None]).sum(1) / np.maximum(cnt, 1)
            mp = (nb[..., None] * pos[None]).sum(1) / np.maximum(cnt, 1)
            acc += self._arr(P, 'ali')[:, None] * np.where(has[:, None], mv - vel, 0) * 1.2
            acc += self._arr(P, 'coh')[:, None] * np.where(has[:, None], (mp - pos) / rad[:, None], 0) * vmax[:, None]
        g = self.goals.copy()
        for t in range(2):
            if self.goal_override[t] is not None:
                g[team == t] = self.goal_override[t]
        acc += self._arr(P, 'goal')[:, None] * _unit(g - pos) * vmax[:, None] * 1.5
        sw = self._arr(P, 'swirl')
        if np.any(sw != 0):
            c = np.array([[P['teams'][t]['zx'] * W, P['teams'][t]['zy'] * H] for t in range(2)])[team]
            r = pos - c
            rn = np.linalg.norm(r, axis=1, keepdims=True) + 1e-6
            acc += sw[:, None] * np.c_[-r[:, 1], r[:, 0]] / rn * vmax[:, None] * 0.8
            acc -= sw[:, None] * 0.4 * (r / rn) * (rn - 0.25 * min(W, H)) / (0.25 * min(W, H)) * vmax[:, None] * 0.3
        self.wander_ang += self.rng.normal(0, 2.0 * dt ** .5, n)
        acc += self._arr(P, 'wander')[:, None] * np.c_[np.cos(self.wander_ang), np.sin(self.wander_ang)] * vmax[:, None]
        acc += np.c_[self._arr(P, 'wind_x'), self._arr(P, 'wind_y')] * vmax[:, None]
        for (a, b, key) in ((0, 1, 'ab'), (1, 0, 'ba')):
            self._interact(acc, dist, diff, P['inter'][key], a, b, P)
        acc[:, 1] *= self._arr(P, 'depth_mob')
        # lissage : evite les changements brusques de direction (moins de chaos)
        sm = self._arr(P, 'smooth')
        k = np.where(sm > 0.001, 1 - np.exp(-dt / np.maximum(sm * 0.6, 1e-3)), 1.0)
        acc = self.acc_prev + (acc - self.acc_prev) * k[:, None]
        self.acc_prev = acc.copy()
        # ancrage : rappel vers le point d'origine au-dela du rayon de liberte
        hm = self._arr(P, 'home')
        if np.any(hm > 0):
            d = self.home - pos
            dn = np.linalg.norm(d, axis=1, keepdims=True) + 1e-6
            r0 = self._arr(P, 'home_r')[:, None] / 100 * H
            over = np.maximum(dn - r0, 0)
            acc = acc + hm[:, None] * (d / dn) * (over / (0.06 * H)) * vmax[:, None]
        m = 0.04 * W
        for t in range(2):
            mt = (team == t) & al
            if mt.any() and P['teams'][t]['boundary'] == BOUNDARIES[0]:
                lo = np.array([-m, 0.30 * H])
                hi = np.array([W + m, H * 1.02])
                p = pos[mt]
                acc[mt] += (np.where(p < lo, lo - p, 0) + np.where(p > hi, hi - p, 0)) / (0.1 * H) * vmax[mt][:, None] * 2
        an = np.linalg.norm(acc, axis=1, keepdims=True) + 1e-6
        acc *= np.minimum(1.0, self._arr(P, 'force')[:, None] * vmax[:, None] / an)
        vel += acc * dt * 3.0
        vel *= (1 - np.minimum(0.9, self._arr(P, 'damp') * dt))[:, None]
        sp = np.linalg.norm(vel, axis=1, keepdims=True) + 1e-6
        vel *= np.minimum(1, vmax[:, None] / sp)
        vel[frozen | ~al] = 0
        pos += vel * dt
        for t in range(2):
            mt = team == t
            bnd = P['teams'][t]['boundary']
            if bnd == BOUNDARIES[1]:
                for k, (lo, hi) in enumerate([(0, W), (0.25 * H, H)]):
                    lw = mt & (pos[:, k] < lo)
                    hg = mt & (pos[:, k] > hi)
                    pos[lw, k] = lo
                    vel[lw, k] *= -1
                    pos[hg, k] = hi
                    vel[hg, k] *= -1
            elif bnd == BOUNDARIES[2]:
                pos[mt, 0] = (pos[mt, 0] + m) % (W + 2 * m) - m
                pos[mt, 1] = np.clip(pos[mt, 1], 0.2 * H, 1.05 * H)
        self._capture(dist, P)
        self.slow = np.minimum(1.0, self.slow + dt / max(P['cap_hold'], 0.2))
        rate = P['anim_speed'] * self.rate_rand * self._arr(P, 'anim_mult')
        if P['sync_speed']:
            rate = rate * np.clip(np.linalg.norm(vel, axis=1) / (0.08 * H), 0.15, 2.5)
        rate = np.where(frozen, rate * 0.15, rate)
        self.phase += dt * rate
        self.size_mult = self._arr(P, 'size_mult')
        self.t += dt
        if P['pos_stepped']:
            s = int(math.floor(self.t * P['cadence']))
            if s != self.disp_step:
                self.disp_step = s
                self.disp = pos + (self.rng.normal(0, P['boil'], pos.shape) if P['boil'] > 0 else 0)
        else:
            self.disp = pos

    def _interact(self, acc, dist, diff, R, a, b, P):
        mode = R['mode']
        if mode == PAIR_MODES[0]:
            return
        IA = np.where((self.team == a) & self.alive)[0]
        IB = np.where((self.team == b) & self.alive)[0]
        if len(IA) == 0 or len(IB) == 0:
            return
        H = self.H
        s = R['strength']
        rg = R['range'] / 100 * H
        pos, vel = self.pos, self.vel
        dAB = dist[np.ix_(IA, IB)]
        j = np.argmin(dAB, axis=1)
        dmin = dAB[np.arange(len(IA)), j]
        tgt = pos[IB[j]] + vel[IB[j]] * R['lead']
        vm = self._vmax[IA][:, None]
        if mode == PAIR_MODES[1]:
            acc[IA] += s * _unit(tgt - pos[IA]) * vm * (dmin < rg)[:, None]
        elif mode == PAIR_MODES[2]:
            ld = self.leader[b]
            if ld >= 0 and self.alive[ld]:
                lt = pos[ld] + vel[ld] * R['lead']
                d = np.linalg.norm(pos[IA] - pos[ld], axis=1)
                acc[IA] += s * _unit(lt - pos[IA]) * vm * (d < rg)[:, None]
            else:
                acc[IA] += s * _unit(tgt - pos[IA]) * vm * (dmin < rg)[:, None]
        elif mode == PAIR_MODES[3]:
            ld = self.leader[b]
            C = pos[ld] if (len(IB) == 1 and ld >= 0) else pos[IB].mean(0)
            ang = 2 * np.pi * np.arange(len(IA)) / len(IA) + self.t * 0.25
            gp = C + max(rg * 0.45, 20) * np.c_[np.cos(ang), np.sin(ang)]
            acc[IA] += s * _unit(gp - pos[IA]) * vm
        elif mode == PAIR_MODES[4]:
            f = np.clip(1 - dmin / max(rg, 1), 0, 1)
            acc[IA] += s * _unit(pos[IA] - tgt) * vm * (dmin < rg)[:, None] * (1 + f[:, None])
        elif mode == PAIR_MODES[5]:
            near = dAB < rg
            dd = diff[np.ix_(IA, IB)]
            sv = ((dd / dAB[..., None]) * (rg / dAB)[..., None] * near[..., None]).sum(1)
            sn = np.linalg.norm(sv, axis=1, keepdims=True) + 1e-6
            acc[IA] += s * sv / sn * np.minimum(sn, 3.0) * vm
        elif mode == PAIR_MODES[6]:
            near = dAB < rg
            c = near.sum(1, keepdims=True)
            mp = np.where(c > 0, (near[..., None] * pos[IB][None]).sum(1) / np.maximum(c, 1), pos[IA])
            mv = np.where(c > 0, (near[..., None] * vel[IB][None]).sum(1) / np.maximum(c, 1), vel[IA])
            acc[IA] += s * (0.5 * _unit(mp - pos[IA]) * vm + (mv - vel[IA]) * 1.2)
        elif mode == PAIR_MODES[7]:
            xl = P['teams'][a]['zx'] * self.W
            gp = np.c_[np.full(len(IA), xl), tgt[:, 1]]
            far = np.linalg.norm(gp - pos[IA], axis=1) > 0.03 * H
            acc[IA] += s * _unit(gp - pos[IA]) * vm * far[:, None]

    def _capture(self, dist, P):
        r = P['cap_radius'] / 100 * self.H
        for (a, b, key) in ((0, 1, 'cap_ab'), (1, 0, 'cap_ba')):
            eff = P[key]
            if eff == CAPTURE[0]:
                continue
            IA = np.where((self.team == a) & self.alive)[0]
            IB = np.where((self.team == b) & self.alive)[0]
            if len(IA) == 0 or len(IB) == 0:
                continue
            d = dist[np.ix_(IB, IA)].min(1)
            hit = IB[d < r]
            if len(hit) == 0:
                continue
            if eff == CAPTURE[1]:
                self.slow[hit] = 0.3
            elif eff == CAPTURE[2]:
                self.frozen_until[hit] = np.maximum(self.frozen_until[hit], self.t + P['cap_hold'])
            elif eff == CAPTURE[3]:
                self.alive[hit] = False
            elif self.pool_idx[a]:
                self.team[hit] = a
                self.seqs[hit] = self.rng.choice(self.pool_idx[a], len(hit))
                for t in range(2):
                    mk = (self.team == t) & self.alive
                    self.leader[t] = int(np.argmax(mk)) if mk.any() else -1


def frame_index(mode, tau, n, src_fps, cadence, rr):
    """tau : temps source (s) de chaque individu (tableau). -> indices d'image dans la boucle."""
    if n <= 1:
        return np.zeros(len(tau), int)
    if mode == ANIM_MODES[0]:
        return np.floor(tau * src_fps).astype(int) % n
    if mode == ANIM_MODES[1]:
        step = max(1, int(round(src_fps / max(cadence, 1))))
        return (np.floor(tau * cadence).astype(int) * step) % n
    if mode == ANIM_MODES[2]:
        p = np.floor(tau * src_fps).astype(int) % (2 * (n - 1))
        return np.where(p < n, p, 2 * (n - 1) - p)
    if mode == ANIM_MODES[3]:
        k = np.floor(tau * cadence).astype(np.int64)
        return ((k * 2654435761 + (rr * 1e6).astype(np.int64)) % 7919) % n
    # pose figee : une pose par individu, balancement lent
    base = (rr * 1e5).astype(int) % n
    sway = np.floor(2 * np.sin(tau * 2.0)).astype(int)
    return (base + sway) % n


def _team(**kw):
    return kw


SCENARIOS = {
    'Libre (aucune interaction)': dict(),
    'Un manifestant, tous les CRS a ses trousses': dict(
        teams=[_team(n=40, place=PLACEMENTS[0], zx=0.22, zw=0.3, speed=15, boundary='Rebond', ali=1.0, coh=0.3, wander=0.2),
               _team(n=1, place=PLACEMENTS[1], zx=0.6, zw=0.2, zh=0.2, speed=16, boundary='Rebond', wander=0.5, sep=0, ali=0, coh=0)],
        inter=dict(ab=dict(mode=PAIR_MODES[2], strength=2.4, range=250, lead=0.6),
                   ba=dict(mode=PAIR_MODES[4], strength=2.8, range=45, lead=0.0)), cap_ab=CAPTURE[1]),
    'Un CRS poursuivi par la foule': dict(
        teams=[_team(n=1, place=PLACEMENTS[1], zx=0.6, zw=0.2, zh=0.2, speed=16, boundary='Rebond', wander=0.5, sep=0, ali=0, coh=0),
               _team(n=40, place=PLACEMENTS[0], zx=0.22, zw=0.3, speed=15, boundary='Rebond', ali=1.0, coh=0.3, wander=0.2)],
        inter=dict(ab=dict(mode=PAIR_MODES[4], strength=2.8, range=45, lead=0.0),
                   ba=dict(mode=PAIR_MODES[2], strength=2.4, range=250, lead=0.6)), cap_ba=CAPTURE[1]),
    'Poursuite de groupe (CRS contre manifestants)': dict(
        teams=[_team(n=45, place=PLACEMENTS[5], zx=0.3, zw=0.2, speed=15, boundary='Rebond', gx=-1, ali=1.2, coh=0.3),
               _team(n=55, place=PLACEMENTS[0], zx=0.62, zw=0.35, speed=14, boundary='Rebond', ali=0.8, coh=0.6, wander=0.5)],
        inter=dict(ab=dict(mode=PAIR_MODES[1], strength=2.0, range=40, lead=0.4),
                   ba=dict(mode=PAIR_MODES[4], strength=2.2, range=30, lead=0.0)), cap_ab=CAPTURE[1]),
    'Nasse : les CRS encerclent la foule': dict(
        teams=[_team(n=44, place=PLACEMENTS[4], zx=0.5, zw=0.9, zh=0.5, speed=10, boundary='Rebond', ali=0.3, coh=0.0, wander=0.1),
               _team(n=50, place=PLACEMENTS[0], zx=0.5, zw=0.3, zh=0.25, speed=5, boundary='Rebond', wander=0.8, coh=0.8, home=1.5, home_r=4.0, damp=2.5)],
        inter=dict(ab=dict(mode=PAIR_MODES[3], strength=1.6, range=55, lead=0.0),
                   ba=dict(mode=PAIR_MODES[4], strength=0.8, range=18, lead=0.0))),
    'Cordon de CRS qui repousse la foule': dict(
        teams=[_team(n=26, place=PLACEMENTS[2], zx=0.38, zw=0.05, zh=0.5, speed=7, boundary='Rebond', ali=0.5, coh=0, sep=1.0, wander=0.1, home=2.5, home_r=2.0, damp=3.0),
               _team(n=70, place=PLACEMENTS[9], zx=0.78, zw=0.35, zh=0.5, speed=10, boundary='Rebond', gx=0.1, goal=0.8, wander=0.6)],
        inter=dict(ab=dict(mode=PAIR_MODES[7], strength=1.6, range=60, lead=0.0),
                   ba=dict(mode=PAIR_MODES[5], strength=1.6, range=18, lead=0.0))),
    'Charge et dispersion': dict(
        teams=[_team(n=40, place=PLACEMENTS[3], zx=0.18, zw=0.1, zh=0.5, rows=4, speed=17, boundary='Libre', gx=0.95, goal=1.8, ali=1.2),
               _team(n=70, place=PLACEMENTS[0], zx=0.6, zw=0.4, speed=16, boundary='Reapparait de l autre cote', wander=0.5)],
        inter=dict(ab=dict(mode=PAIR_MODES[0]), ba=dict(mode=PAIR_MODES[4], strength=3.0, range=30, lead=0.0))),
    'Cortege escorte par des CRS': dict(
        teams=[_team(n=30, place=PLACEMENTS[6], zx=0.5, zw=0.9, zh=0.5, speed=7, boundary='Reapparait de l autre cote', wind_x=1.0, ali=2.0, coh=0.2, wander=0.15, direction=0),
               _team(n=70, place=PLACEMENTS[6], zx=0.5, zy=0.55, zw=0.9, zh=0.3, speed=7, boundary='Reapparait de l autre cote', wind_x=1.0, ali=2.5, coh=0.2, wander=0.15, direction=0)],
        inter=dict(ab=dict(mode=PAIR_MODES[6], strength=1.2, range=45, lead=0.0), ba=dict(mode=PAIR_MODES[0]))),
    'Affrontement : deux camps face a face': dict(
        teams=[_team(n=45, place=PLACEMENTS[2], zx=0.25, zw=0.15, zh=0.45, speed=11, boundary='Rebond', gx=0.75, goal=1.2, ali=1.0),
               _team(n=55, place=PLACEMENTS[2], zx=0.75, zw=0.15, zh=0.45, speed=11, boundary='Rebond', gx=0.25, goal=1.2, ali=1.0, direction=180)],
        inter=dict(ab=dict(mode=PAIR_MODES[1], strength=1.4, range=22, lead=0.3), ba=dict(mode=PAIR_MODES[1], strength=1.4, range=22, lead=0.3))),
    'Contagion : les CRS attrapent et convertissent': dict(
        teams=[_team(n=6, place=PLACEMENTS[1], zx=0.25, zw=0.2, speed=13, boundary='Rebond', ali=0.5, coh=0.2),
               _team(n=70, place=PLACEMENTS[1], zx=0.65, zw=0.5, speed=11, boundary='Rebond', wander=0.9, coh=0.4)],
        inter=dict(ab=dict(mode=PAIR_MODES[1], strength=2.0, range=40, lead=0.3), ba=dict(mode=PAIR_MODES[4], strength=2.0, range=25, lead=0)),
        cap_ab=CAPTURE[4]),
}


def apply_scenario(P, name):
    """Applique un scenario a P (modifie les equipes, interactions, captures). Les champs non cites restent aux valeurs de base."""
    sc = SCENARIOS.get(name, {})
    P['teams'] = [dict(team_defaults(i)) for i in range(2)]
    P['inter'] = dict(ab=inter_defaults(), ba=inter_defaults())
    P['cap_ab'] = sc.get('cap_ab', CAPTURE[0])
    P['cap_ba'] = sc.get('cap_ba', CAPTURE[0])
    for i, d in enumerate(sc.get('teams', [])):
        P['teams'][i].update(dict(MOVE_EXTRA, smooth=0.3))
        P['teams'][i].update(d)
    for k, d in sc.get('inter', {}).items():
        P['inter'][k].update(d)
