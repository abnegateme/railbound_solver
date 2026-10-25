# Normalized cell extraction (M2)

`extract` rectifies every calibrated cell into an oriented square BGR PNG,
then writes a deterministic manifest and contact sheet.

```bash
python3 -m railbound extract \
  --image data/screenshot.png \
  --grid output/grid.json \
  --output-dir output/crops \
  --mode context --mask-mode cell
```

`core` crops exclude borders; `context` includes nearby pixels.
`--mask-mode cell` blacks out contextual pixels outside the target cell. The
manifest stores extraction settings, and the command rejects a source whose
digest differs from the calibrated grid.
