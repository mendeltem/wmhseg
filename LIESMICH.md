# wmhseg -- die Werkbank dazu (deutsch, lokal)

Der englische `README.md` ist fuer das Repository. Diese Datei ist fuer uns.

START: wmhseg --flair FLAIR_brain_biascorr.nii.gz --out wmh.nii.gz

## Wo was liegt
    wmhseg/netz.py           Netz (BAM-U-Net), aus arena.py herausgeloest
    wmhseg/kette.py          die Kette FLAIR -> Maske, Schritt fuer Schritt
    wmhseg/cli.py            der Befehl
    wmhseg/modell/           gewichte.pt (7,7 MB), modell.json (BEREINIGT), sha256.txt
    gleichheit.json          Beweis: gleiche Ausgabe wie arena.py, 12 Challenge-Faelle
    qc_challenge/            die oeffentliche QC-Seite, nur Challenge-Bilder
    ../gleichheit_pruefen.py erzeugt gleichheit.json neu
    ../qc_bauen.py           erzeugt die QC-Seite neu
    ../wmhseg_vorhersagen/   die Vorhersagen dieser 12 Faelle (bleiben lokal)

## Was schon fertig ist
- Paket laeuft: `pip install -e .` gemacht, `wmhseg --version` antwortet, ein echter
  Lauf auf sub-070 liefert 10 482 Voxel. Gegen den Lauf mit GELIEFERTER Hirnmaske:
  Dice 0.9989 -- der Rest ist allein die selbst abgeleitete Maske, nicht das Netz.
- Gleichheit mit der Trainingskette: 12 von 12 Faellen **Dice 1.000000**, CPU,
  gleiche Gewichte, gleiche Hirnmaske. Form und Affine identisch.
- QC-Seite mit echten Challenge-Bildern, drei Spalten, schlechteste Faelle oben.
- `modell.json` im Paket ist bereinigt: die 41 eigenen Fallnamen aus `val_faelle`
  sind draussen, die vollstaendige Fassung liegt unter `../modell/`.

## Was noch fehlt
- **FLAIR+T1**: ein zweikanaliges Modell gibt es noch nicht (Auftraege wh3/wh6).
  Die Kette kann zwei Kanaele, das Werkzeug meldet sauber, dass das Modell einkanalig
  ist. Sobald die Gewichte da sind, reicht ein zweiter Modellordner.
- Selbsttest (`wmhseg --selbsttest`) mit einem mitgelieferten Fall -- steht in wb6.
- Die Gleichheit ist auf CPU geprueft; auf der GPU rechnet die Arena mit
  bfloat16-Autocast, das Werkzeug ebenso. Nicht gegengeprueft.

## Fallstricke, die Zeit gekostet haben
- Normiert wird INNERHALB der Hirnmaske, nicht ueber das Volumen.
- Geschwellt wird ERST nach dem Rueckweg auf das FLAIR-Gitter.
- Der 256er-Ausschnitt haengt am Schwerpunkt der Maske, nicht an der Bildmitte.
