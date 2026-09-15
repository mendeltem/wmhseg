"""Die Kette vom FLAIR zur WMH-Maske -- Schritt fuer Schritt wie im Training.

Jeder Schritt hier hat sein Gegenstueck im Trainingswerkzeug (arena.py). Weicht
einer ab, weicht das Ergebnis ab: die Normierung geschieht INNERHALB der
Hirnmaske, die Ebene wird auf 1,0 mm gebracht, der 256er-Ausschnitt liegt auf dem
Schwerpunkt der Maske, und geschwellt wird ERST nach dem Rueckweg auf das
FLAIR-Gitter. Deshalb steht die Kette geschlossen in einer Datei.
"""
import json
import os

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import binary_fill_holes, center_of_mass, zoom

from . import netz as _netz

KANTE = 256
HIER = os.path.dirname(os.path.abspath(__file__))
MODELLORDNER = os.path.join(HIER, "modell")


def modellpfade(ordner=None):
    o = ordner or os.environ.get("WMHSEG_MODELL") or MODELLORDNER
    return (os.path.join(o, "gewichte.pt"), os.path.join(o, "modell.json"),
            os.path.join(o, "sha256.txt"))


def lade_modell(ordner=None, geraet="cpu", pruefe_summe=True):
    gew, meta, summe = modellpfade(ordner)
    for p in (gew, meta):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{p} fehlt. Die Gewichte gehoeren ins Paket (wmhseg/modell/) "
                f"oder in den Ordner aus --modell / $WMHSEG_MODELL.")
    with open(meta, encoding="utf-8") as f:
        info = json.load(f)
    if pruefe_summe and os.path.exists(summe):
        import hashlib
        with open(gew, "rb") as f:
            ist = hashlib.sha256(f.read()).hexdigest()
        with open(summe, encoding="utf-8") as f:
            soll = f.read()
        if ist not in soll:
            raise ValueError(
                f"sha256 der Gewichte passt nicht zu sha256.txt ({ist[:12]}...). "
                f"Die Datei ist unterwegs veraendert worden -- nicht benutzen.")
    modell = _netz.bau(len(info["eingaben"]))
    modell.load_state_dict(torch.load(gew, map_location="cpu"))
    modell.to(geraet).eval()
    return modell, info


def hirnmaske_aus_flair(fl):
    """Aus einem hirnextrahierten FLAIR: alles > 0, Loecher gefuellt."""
    return binary_fill_holes(fl > 0)


def _resample_inplane(arr, zooms, order=1):
    f = (float(zooms[0]), float(zooms[1]), 1.0)
    return zoom(arr, f, order=order)


def _crop_pad(a2d, x0, y0, k=KANTE):
    h, w = a2d.shape
    sx0, sx1 = max(0, x0), min(h, x0 + k)
    sy0, sy1 = max(0, y0), min(w, y0 + k)
    out = np.zeros((k, k), a2d.dtype)
    out[sx0 - x0:sx1 - x0, sy0 - y0:sy1 - y0] = a2d[sx0:sx1, sy0:sy1]
    return out


def vorbereiten(flair_pfad, t1_pfad=None, maske_pfad=None):
    """FLAIR (und optional T1) lesen, normieren, auf 1,0 mm in der Ebene bringen."""
    bild = nib.load(flair_pfad)
    fl = np.asarray(bild.dataobj, dtype=np.float32)
    zooms = bild.header.get_zooms()[:3]

    if maske_pfad:
        maske_fl = np.asarray(nib.load(maske_pfad).dataobj) > 0.5
        if maske_fl.shape != fl.shape:
            raise ValueError("Die Hirnmaske liegt nicht auf dem FLAIR-Gitter.")
    else:
        maske_fl = hirnmaske_aus_flair(fl)
    if not maske_fl.any():
        raise ValueError("Die Hirnmaske ist leer -- ist das FLAIR wirklich hirnextrahiert?")

    kanaele = [fl]
    if t1_pfad:
        t1 = np.asarray(nib.load(t1_pfad).dataobj, dtype=np.float32)
        if t1.shape != fl.shape:
            raise ValueError("Das T1 liegt nicht auf dem FLAIR-Gitter.")
        kanaele.append(t1)

    maske1 = _resample_inplane(maske_fl.astype(np.float32), zooms, order=0) > 0.5
    X = np.zeros(maske1.shape + (len(kanaele),), dtype=np.float32)
    for i, a in enumerate(kanaele):
        mu, sig = a[maske_fl].mean(), a[maske_fl].std()
        if sig < 1e-8:
            sig = 1.0
        X[..., i] = _resample_inplane(np.where(maske_fl, (a - mu) / sig, 0.0), zooms, order=1)
    X = np.where(maske1[..., None], X, 0.0)

    cx, cy, _ = center_of_mass(maske1)
    return {
        "X": np.moveaxis(X, 2, 0),
        "maske_fl": np.moveaxis(maske_fl, 2, 0),
        "affine": bild.affine, "zooms": zooms, "fl_shape": fl.shape,
        "x0": int(round(cx - KANTE / 2)), "y0": int(round(cy - KANTE / 2)),
        "anteil_ueber_null": float((fl > 0).mean()),
    }


def wahrscheinlichkeiten(modell, X, x0, y0, geraet="cpu"):
    """X (z,x1,y1,c) -> Wahrscheinlichkeiten (z,x1,y1), Schnitt fuer Schnitt."""
    z, x1, y1, c = X.shape
    w = np.zeros((z, x1, y1), np.float32)
    amp = (geraet != "cpu")
    with torch.no_grad():
        for zz in range(z):
            if X[zz].max() == 0:
                continue
            patch = np.stack([_crop_pad(X[zz, :, :, ci], x0, y0) for ci in range(c)],
                             axis=0)[None].astype(np.float32)
            t = torch.from_numpy(patch).to(geraet)
            if amp:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    wz = torch.sigmoid(modell(t).float())
            else:
                wz = torch.sigmoid(modell(t).float())
            wx = wz[0, 0].cpu().numpy()
            sx0, sx1 = max(0, x0), min(x1, x0 + KANTE)
            sy0, sy1 = max(0, y0), min(y1, y0 + KANTE)
            w[zz, sx0:sx1, sy0:sy1] = wx[sx0 - x0:sx1 - x0, sy0 - y0:sy1 - y0]
    return w


def rueckweg(w1, d):
    """Wahrscheinlichkeiten vom 1-mm-Gitter zurueck auf das FLAIR-Gitter."""
    z, x1, y1 = w1.shape
    fx, fy, fz = d["fl_shape"]
    out = np.zeros((fz, fx, fy), dtype=np.float32)
    for zz in range(min(z, fz)):
        r = zoom(w1[zz], (1.0 / d["zooms"][0], 1.0 / d["zooms"][1]), order=1)
        rh, rw = r.shape
        ox, oy = (fx - rh) // 2, (fy - rw) // 2
        r0, r1 = max(0, -ox), min(rh, fx - ox)
        c0, c1 = max(0, -oy), min(rw, fy - oy)
        out[zz, max(0, ox):max(0, ox) + (r1 - r0),
            max(0, oy):max(0, oy) + (c1 - c0)] = r[r0:r1, c0:c1]
    if out.shape != (fz, fx, fy):
        raise ValueError(f"Rueckweg liefert {out.shape}, erwartet {(fz, fx, fy)}")
    return out


def segmentieren(flair, t1=None, maske=None, modell=None, info=None,
                 geraet="cpu", schwelle=None, modellordner=None):
    """FLAIR -> (Maske uint8 (x,y,z), Wahrscheinlichkeiten (x,y,z), Affine, Info)."""
    if modell is None:
        modell, info = lade_modell(modellordner, geraet)
    if t1 is not None and len(info["eingaben"]) == 1:
        raise ValueError(
            "Dieses Modell erwartet nur FLAIR. Fuer FLAIR+T1 braucht es ein "
            "zweikanaliges Modell -- es ist hier nicht hinterlegt (--modell).")
    if t1 is None and len(info["eingaben"]) == 2:
        raise ValueError("Dieses Modell erwartet FLAIR UND T1 (--t1).")
    d = vorbereiten(flair, t1, maske)
    w1 = wahrscheinlichkeiten(modell, d["X"], d["x0"], d["y0"], geraet)
    w_fl = rueckweg(w1, d)
    s = float(schwelle if schwelle is not None else info["Schwelle"])
    pred = ((w_fl >= s) & d["maske_fl"]).astype(np.uint8)
    return (np.moveaxis(pred, 0, 2), np.moveaxis(w_fl, 0, 2), d["affine"],
            {"schwelle": s, "anteil_ueber_null": d["anteil_ueber_null"]})
