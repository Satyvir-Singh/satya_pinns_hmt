# Provenance of the Ghia et al. (1982) Re = 100 reference data

**Reference.** U. Ghia, K. N. Ghia and C. T. Shin, "High-Re solutions for incompressible flow
using the Navier–Stokes equations and a multigrid method", *Journal of Computational Physics*
48 (1982) 387–411. DOI 10.1016/0021-9991(82)90058-4.

| File | Content |
|---|---|
| `ghia1982_re100_table1_u_vertical_centerline.csv` | Table I, Re = 100 column: u along x = 0.5 (17 points, y descending) |
| `ghia1982_re100_table2_v_horizontal_centerline.csv` | Table II, Re = 100 column: v along y = 0.5 (17 points, x descending) |

## Verification status: UNVERIFIED against the original article

On 2026-10-11 the original article could not be accessed: the publisher page
(sciencedirect.com/science/article/pii/0021999182900584) returned HTTP 403, and no open copy was
found. **The values have therefore not been checked against the printed tables.** All validation
statements in this project carry that caveat.

The values were transcribed from two independent secondary sources and cross-checked
programmatically. All 34 coordinates and 34 velocities agree exactly in both sources (max |difference| = 0):

* **S1:** GitHub user "ivan-pi", gists `ghiau.txt` (Table I) and `ghiav.txt` (Table II), created
  2019-04-18. Both list all Reynolds-number columns.
  <https://gist.github.com/ivan-pi/3e9326d18a366ffe6a8e5bfda6353219>,
  <https://gist.github.com/ivan-pi/caa6c6737d36a9140fbcf2ea59c78b3c>.
  Verbatim snapshots are in `sources/`:
  * SHA-256 `51b65a48…2f3b54` (ghiau)
  * SHA-256 `9589dde4…cd14d0` (ghiav)
* **S2:** Uppsala University degree project (Examensarbete 15 hp, June 2022), DiVA
  `diva2:1668016`, Tables 1–2. Source: <https://uu.diva-portal.org/smash/get/diva2:1668016/FULLTEXT01.pdf>.
  The PDF downloaded on 2026-10-11 has SHA-256 `52403c7b…157257`. It was not stored here.
  Its text was extracted with `pypdf`.

A third source, a `reference_v.txt` file on nextjournal.com, was seen only in a search snippet.
It also gives 0.9531 → −0.08864 and 0.9453 → −0.10313, but it prints x = 0.9606 for 0.9609.
That source was not used.

### Known discrepancy in circulating copies
Some copies of Table II in circulation assign −0.10313 to x = 0.9531 and −0.08864 to x = 0.9453.
That ordering gives a non-monotone profile. S1 and S2 both give the monotone ordering used here.
Without the original, which ordering is correct cannot be confirmed. The affected points are 2 of
17 v-points, and the values differ by 0.0145.

### Coordinate convention
Ghia's grid is a uniform 129 × 129 mesh. The printed coordinates are grid nodes k/128 rounded to
4 decimals; for example, 0.9766 corresponds to 125/128 = 0.9765625. The PINN is evaluated
directly at the printed coordinates. The rounding (≤ 5e-5) is far below any relevant error level.

### Use restrictions
These data are used **only** for post-training validation in `cavity_pinn/evaluate.py` and
`cavity_pinn/summarize.py`. They are never used in training, sampling, loss weighting, early
stopping or checkpoint selection. `tests/test_benchmark.py` asserts that the training modules
never reference them.
