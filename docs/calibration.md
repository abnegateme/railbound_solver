# Manual grid calibration (M1)

`calibrate` converts three adjacent cell centres into a portable affine
`grid.json` and an annotated PNG. Use it for reproducible batch work or when
automatic detection needs correction.

```bash
python3 -m railbound calibrate \
  --image data/screenshot.png \
  --rows 8 --cols 10 \
  --origin 120 240 \
  --next-col 172 240 \
  --next-row 120 292 \
  --grid-output output/grid.json \
  --overlay-output output/annotated_grid.png
```

`origin`, `next-col`, and `next-row` are the centres of `(0, 0)`, `(0, 1)`,
and `(1, 0)`. Pixels use `(x, y)`; cells use `(row, col)`. The saved grid
records source dimensions and an optional SHA-256 digest, never a local path.
