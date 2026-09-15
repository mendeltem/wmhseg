# wmhseg

WMH segmentation (white matter hyperintensities) on FLAIR, in one command.
A 2D U-Net with Bottleneck Attention Modules (BAM), shipped with its weights.

**See it before you install it: [the QC page](https://mendeltem.github.io/wmhseg/)** —
reference, BIANCA and `wmhseg` side by side on public challenge data, worst cases first.

```bash
pip install git+https://github.com/mendeltem/wmhseg
wmhseg --flair FLAIR_brain_biascorr.nii.gz --out wmh.nii.gz
```

The weights (7.7 MB) are part of the package. There is no separate download step.

## Input: what the model expects

The FLAIR must be **brain-extracted and bias-corrected**. That is exactly what the
model was trained on, and feeding it a raw FLAIR gives a plausible-looking but wrong
result. `wmhseg` checks the input: if more than 40 % of the volume is greater than
zero it stops with a message instead of computing nonsense (`--ungeprueft` overrides).

The tool does **no** preprocessing of its own -- no HD-BET, no FSL, no external
programs. It is plain Python (torch, nibabel, numpy, scipy). Prepare the image with
whatever pipeline you already trust; a common one is HD-BET followed by N4/FAST bias
correction.

```bash
wmhseg --flair flair.nii.gz --out wmh.nii.gz          # brain mask derived from the FLAIR
wmhseg --flair flair.nii.gz --maske mask.nii.gz --out wmh.nii.gz
wmhseg --flair flair.nii.gz --out wmh.nii.gz --prob prob.nii.gz --cpu
wmhseg --version
```

Output is a `uint8` NIfTI (0/1) on the grid and affine of the input FLAIR.

## What it does, step by step

1. Brain mask: from `--maske`, or `binary_fill_holes(flair > 0)`.
2. z-normalisation **inside the brain mask**, zero outside.
3. In-plane resampling to 1.0 mm, slice direction untouched.
4. A 256x256 crop per axial slice, centred on the centre of mass of the mask.
5. The network, slice by slice.
6. Back to the FLAIR grid, restricted to the brain mask, **then** thresholded (0.40).

Steps 2 and 6 are the ones people get wrong when reimplementing: normalising over the
whole volume, or thresholding before returning to the original grid, both shift the
result measurably.

## Does it compute what it was trained to compute?

`gleichheit.json` in this repository is the evidence: on 12 cases of the public
challenge test set the output of `wmhseg` is **identical** to the output of the
training pipeline -- Dice 1.000000, same shape, same affine, case by case.

Deriving the brain mask from the FLAIR instead of passing one with `--maske` changes
the result slightly: on `sub-070` the two masks agree at Dice 0.9989. Pass the mask you
used for brain extraction if you want the exact same result twice.

## Quality control

**[mendeltem.github.io/wmhseg](https://mendeltem.github.io/wmhseg/)** (source: `docs/index.html`)
shows, per case, the same axial slice three times
with the same grey window: reference, BIANCA (the established method here), and
`wmhseg`. Sorted worst-first. Everything shown comes from the public
[WMH Segmentation Challenge 2017](https://wmh.isi.uu.nl/) test set.

On that 12-case sample the mean Dice is 0.710 for `wmhseg` and 0.677 for BIANCA.
This is a sample for looking at, not a benchmark result.

## Training data, and what that means for you

The model was trained on 201 cases: 60 from the public WMH challenge training set and
141 from three in-house clinical cohorts (1.5 T and 3 T, several scanner models).
No case identifiers from the in-house cohorts are contained in this repository.

The honest limitation: performance was measured on this data, and on an unseen public
domain it is lower than on the training domain. Treat it as a research tool, validate
it on your own data before you rely on it, and do not use it for clinical decisions.

## Model card, short form

| | |
|---|---|
| Architecture | U-Net 2D, channels 32/64/128/256, BAM after every conv block |
| Input | 1 channel, FLAIR, brain-extracted and bias-corrected |
| Patch | 256x256 axial slices on a 1.0 mm in-plane grid |
| Threshold | 0.40, chosen on a validation split (not on the test set) |
| Weights | `wmhseg/modell/gewichte.pt`, 7.7 MB, sha256 checked at start-up |
| Trained | 2026-09-12, PyTorch 2.6 |

## Licence and citation

Code: MIT. The challenge images shown in the QC page belong to the WMH Segmentation
Challenge (Kuijf et al., IEEE TMI 2019) -- cite them if you use that data.
