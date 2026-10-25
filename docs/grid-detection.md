# Automatic grid detection (M3)

`detect` proposes an affine grid from regularly spaced visible boundaries. It
writes a grid only when its strongest candidate reaches the configured
confidence; otherwise inspect the overlay and calibrate manually.

```bash
python3 -m railbound detect \
  --image data/screenshot.png \
  --rows 8 --cols 10 \
  --grid-output output/grid.json \
  --overlay-output output/detected_grid.png
```

`--rows` and `--cols` are optional but help on visually busy boards. The
default threshold is `0.70`; exit status `2` means no candidate was accepted.
Use `--min-cell-size` and `--max-cell-size` only when display scale is known.
