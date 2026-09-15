"""wmhseg -- WMH-Segmentierung auf FLAIR, ein Befehl.

    wmhseg --flair FLAIR_brain_biascorr.nii.gz --out wmh.nii.gz

Das Bild muss hirnextrahiert und bias-korrigiert sein: genau darauf ist das
Modell trainiert. Das Werkzeug selbst verarbeitet nichts vor -- es rechnet, was
es bekommt, und sagt es, wenn die Eingabe nicht danach aussieht.
"""
import argparse
import os
import sys

import nibabel as nib

from . import kette

SCHWELLE_ROH = 0.40   # Anteil Voxel > 0 im Volumen, ab dem es nicht hirnextrahiert aussieht


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="wmhseg",
        description="WMH-Segmentierung auf hirnextrahiertem, bias-korrigiertem FLAIR.")
    p.add_argument("--flair", help="FLAIR (hirnextrahiert, bias-korrigiert)")
    p.add_argument("--t1", help="T1 auf dem FLAIR-Gitter (nur mit zweikanaligem Modell)")
    p.add_argument("--maske", help="Hirnmaske; ohne sie wird sie aus dem FLAIR abgeleitet")
    p.add_argument("--out", help="Ausgabedatei (NIfTI, uint8 0/1)")
    p.add_argument("--prob", help="zusaetzlich die Wahrscheinlichkeiten hierhin schreiben")
    p.add_argument("--schwelle", type=float, help="Schwelle ueberschreiben (Vorgabe aus modell.json)")
    p.add_argument("--modell", help="Ordner mit gewichte.pt, modell.json, sha256.txt")
    p.add_argument("--cpu", action="store_true", help="auf der CPU rechnen")
    p.add_argument("--ungeprueft", action="store_true",
                   help="die Warnung zur Vorverarbeitung uebergehen")
    p.add_argument("--version", action="store_true", help="Modell und Schwelle zeigen")
    a = p.parse_args(argv)

    if a.version:
        _, info = kette.lade_modell(a.modell, "cpu")
        print(f"wmhseg   Modell {info.get('netz')}   Eingaben {info.get('eingaben')}   "
              f"Schwelle {info.get('Schwelle')}   trainiert {info.get('Datum')}")
        return 0

    if not a.flair or not a.out:
        p.error("--flair und --out werden gebraucht (oder --version).")

    import torch
    geraet = "cpu" if (a.cpu or not torch.cuda.is_available()) else "cuda"

    modell, info = kette.lade_modell(a.modell, geraet)
    maske8, prob, affine, stand = kette.segmentieren(
        a.flair, a.t1, a.maske, modell, info, geraet, a.schwelle)

    if stand["anteil_ueber_null"] > SCHWELLE_ROH and not a.ungeprueft:
        print(f"wmhseg: {stand['anteil_ueber_null']:.0%} des Volumens sind groesser null -- "
              f"das sieht NICHT hirnextrahiert aus. wmhseg erwartet ein "
              f"hirnextrahiertes, bias-korrigiertes FLAIR. Mit --ungeprueft trotzdem rechnen.",
              file=sys.stderr)
        return 2

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    nib.save(nib.Nifti1Image(maske8, affine), a.out)
    if a.prob:
        nib.save(nib.Nifti1Image(prob.astype("float32"), affine), a.prob)
    print(f"{a.out}  ({int(maske8.sum())} Voxel, Schwelle {stand['schwelle']}, {geraet})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
