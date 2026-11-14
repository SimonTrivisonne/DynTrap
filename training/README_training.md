# Training Modules: Event Labelling and Model Training

Two scripts that turn VASP molecular-dynamics (MD) output into training data for
machine-learning models that predict **polaron hopping** between Ti sites, and train those models.

| Script | Purpose |
|---|---|
| `label_events.py` | Parses OUTCAR files, tracks the polaron(s) through the MD run, finds *events of interest* (EoI = polaron leaves one Ti and localises on another), builds feature tables and labels, writes `.pkl` files. |
| `train_models.py` | Loads those `.pkl` files, scales the features and trains the neural networks. |

Two models are produced:

| Model | Question it answers | Task | Data from `label_events.py` | `ml_type` in `train_models.py` | Target | Loss |
|---|---|---|---|---|---|---|
| **INSTA** | Is the polaron on this Ti currently at rest or about to diffuse? | Classification (`REST` / `DIFFUSION`) | `*_clas.pkl` | `"CLASSIFICATION"` | `REST`, `DIFFUSION` | binary cross-entropy |
| **HOSTA** | Once the polaron is unstable, which Ti becomes the next host? | Regression of the site magnetisation `MAG` | `*_reg.pkl` | `"REGRESSION"` | `MAG` | MAE |

HOSTA predicts `|MAG|` for every candidate Ti site. The site with the largest predicted value is the
predicted next host. The scripts only train the regressor; the argmax step belongs in your downstream code.

---

## 1. Pipeline

```
 OUTCAR_n  ─┬─►  extract_s1a()          candidate Ti sites + their neighbours
            ├─►  awk                    lattice lengths, positions + forces per MD step
            ├─►  process_pos()          pairwise distance array  dist[step, atom, atom]
 MAGNETIZATIONS_OUTCAR_n.dat ─► mag[step, ti, 1]
                         │
                         ▼
               find_polaron_paths()      two largest |MAG| Ti per step  → polaron trajectory
                         │
          ┌──────────────┴───────────────┐
          ▼                              ▼
  generate_sitefinding_input()     find_EoI() → generate_clssification()
                                          → generate_stability_input()
          │                              │
   data_OUTCAR_*_reg.pkl           data_OUTCAR_*_clas.pkl
          │                              │
          └──────────── train_models.py ─┘
                 HOSTA (regression)   INSTA (classification)
```

## 2. Requirements

- Python 3.9+
- `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `tqdm`, `tensorflow` (Keras)
- `awk` on the PATH. `label_events.py` calls it via `subprocess`, so on Windows use WSL or Git Bash.
- Two helper modules imported by `label_events.py` that are **not part of the two scripts**:
  - `create_ref_dist_data.py` provides `REF_TI_DIST`, `REF_O_B_DIST`, `REF_O_T_DIST`, `REF_O_AP_DIST` (ideal bond lengths used as reference).
  - `process_pos.py` provides `proces_pos(outcar_path, raw_pos)`, which returns the distance array `dist`.

```bash
pip install numpy pandas scipy scikit-learn matplotlib tqdm tensorflow
```

## 3. Expected data layout

```
data/
├── MD700K/                         OUTCAR.0, OUTCAR.2, OUTCAR.3-nodefail, ...
│                                   MAGNETIZATIONS_OUTCAR_<id>.dat       (UTF-16)
├── database_1VO-700K_2017/         OUTCAR_1 ... OUTCAR_28
│                                   MAGNETIZATIONS_OUTCAR_<id>.dat       (UTF-8)
└── labeled_data_v3/                output directory (must exist before running)
```

### `MAGNETIZATIONS_OUTCAR_<id>.dat`

Plain text with two space-separated columns, `MAG STEP`, one row per ion per MD step, ions in OUTCAR order.
The first `TI_IONS` rows of each step are the Ti atoms. The extraction is commented out in `label_events.py`
(awk block "uncomment this to create the mag numpy array ...") and reads the 5th column
of the `magnetization (x)` table.

### What is read directly from the OUTCAR

| Quantity | How | Used for |
|---|---|---|
| Neighbour table (`ion position nearest neighbor table`) | `extract_s1a()` | candidate sites and their 6 O and 2 Ti neighbours |
| `length of vectors` | awk | periodic box (`lattice_vec`) |
| `POSITION ... TOTAL-FORCE` blocks | awk | `X Y Z Fx Fy Fz` per atom and step |

The code assumes an orthorhombic cell (see `distance_3d_periodic`).

## 4. Step 1: `label_events.py`

### 4.1 Run

Edit the `outcars` list in `__main__` and run:

```bash
python label_events.py
```

```python
outcars = [
    {'path': 'data/MD700K/',                  'outcar_ids': [0, 2, 3, 4, 5, 6]},
    {'path': 'data/database_1VO-700K_2017/',  'outcar_ids': list(range(1, 29))},
]
```

The OUTCAR file name and the output file name are derived from `path` in the `if/elif` blocks near the end of the script.

### 4.2 What happens per OUTCAR

1. **Candidate sites.** `extract_s1a()` selects the "S1A" Ti atoms by fractional-coordinate windows
   (`y ≈ 0.999` or `0.498`, `z ≈ 0.364`). For each one it builds a neighbour dictionary:
   `t1/t2` (O above), `b1/b2` (O below), `l/r` (O at the same height), `ti1/ti2` (the two Ti neighbours).
   The OUTCAR neighbour list is sorted by distance, so the direction is recovered by comparing z components.
   Missing Ti neighbours are filled in by x/y matching.
2. **Positions, forces and distances.** Read via awk, then `proces_pos()` gives `dist`.
   `dist[step, ti, 539]` is used as "distance of Ti to its T = 0 position".
3. **Polaron tracking.** `find_polaron_paths()` takes the two Ti with the largest `|MAG|` at each step
   (two excess electrons from one O vacancy). Single-step flicker is removed:
   if the IDs at `t-1` and `t+1` agree but `t` differs, `t` is overwritten.
4. **HOSTA data.** `generate_sitefinding_input()` creates one row per S1A Ti and MD step.
   It keeps only steps where a polaron sits on an S1A Ti (`valid_timesteps`). Target is `|MAG|`.
5. **INSTA data.**
   - `find_EoI(mag, sites, th_a=0.2, th_b=0.05)`: an EoI is the step where `|MAG|` has dropped from above `th_a`
     and stays below `th_b` for 4 consecutive steps. The end-of-simulation case is handled separately.
   - `generate_clssification(EoI, mag, th=0.75)`: for each EoI `e`, `tau` is the last step before `e` with `|MAG| > th`.
     Steps `tau…e` are labelled `DIFFUSION = 1`. Events are discarded if `|MAG|` dips below 0.05 in that window.
     Steps between the previous EoI (or the start) and `tau` with `|MAG| > 0.3` are labelled `REST = 1`
     (the first 10 per segment are skipped; `EoI_ID` is negative for rest segments).
   - `generate_stability_input()` attaches the features below to those rows.

### 4.3 Features (identical definitions in both tables)

| Feature | Meaning |
|---|---|
| `Ti_1`, `Ti_2` | distance to the two Ti neighbours minus `REF_TI_DIST` |
| `O_1`, `O_2` / `O_3`, `O_4` / `O_5`, `O_6` | distance to the O neighbours (`t1,t2` / `b1,b2` / `l,r`) minus `REF_O_T_DIST` / `REF_O_B_DIST` / `REF_O_AP_DIST` |
| `AVG_Ti`, `AVG_O_T`, `AVG_O_B`, `AVG_O_AP` | pair averages of the above |
| `Fx`, `Fy`, `Fz` | force on the Ti atom |
| `DIST_VACANCY` | periodic distance to the (hard-coded) O-vacancy position |
| `DIST_POL1`, `DIST_POL2` | periodic distance to the two tracked polaron positions at the same step |
| `d*` | first time difference within an event segment (1 row dropped) |
| `dd*` | second time difference (2 rows dropped) |

`Ti_0` / `dTi_0` (displacement from the T = 0 position) are computed for INSTA but not used in `train_models.py`.

### 4.4 Output

`data/labeled_data_v3/data_OUTCAR_<id>[_2017|_4]_{reg|clas}.pkl`, pandas DataFrames.

- `*_clas.pkl` columns: `TI, EoI_ID, TIME, MAG, REST, DIFFUSION` + features
- `*_reg.pkl` columns: `TI, TIME, EoI (=0), MAG` + features

## 5. Step 2: `train_models.py`

### 5.1 Run

Select the model at the top of `__main__`:

```python
ml_type = "REGRESSION"        # HOSTA   (reads *_reg.pkl)
# ml_type = "CLASSIFICATION"  # INSTA   (reads *_clas.pkl)
```

Check `OUTCARFILE_ID` and `OUTCARFILE_ID_2017` so they match what `label_events.py` wrote, then run:

```bash
python train_models.py
```

### 5.2 What it does

- Concatenates all pickles and picks `feature_list`.
  - HOSTA uses the base, `d*` and `DIST_*` features (no `dd*`).
  - INSTA uses the base, `d*` and `dd*` features (no `DIST_*`).
- Splits 80/20 **chronologically** (`shuffle=False`).
- Scales with `PowerTransformer` (fit on train only).
- **HOSTA:** trains three MLPs in a loop (`reference_model_regression(X, 0|1|2)`, deep to shallow), MAE loss, Adam 1e-3,
  batch 64, up to 2000 epochs, early stopping on `val_mean_absolute_error` (patience 10).
- **INSTA:** one MLP (64-128, dropout 0.5, 2 sigmoid outputs), binary cross-entropy, batch 16, early stopping on `val_loss`
  (patience 30), with balanced class weights from `generate_class_weights`.
- The script ends in `breakpoint()` so you can inspect the histories. **Nothing is saved by default.**

### 5.3 Saving and using the models

Uncomment and adapt the save line, and also persist the scaler, because inference needs it:

```python
import joblib
mlp.save("models/insta.h5")                      # or mlps[i].save(...) for HOSTA
joblib.dump(scaler, "models/insta_scaler.joblib")
```

Typical inference (derived from `check_data_classification`):

- **INSTA:** `p = mlp.predict(scaler.transform(df[feature_list]))`, then `p[:,0]` is `REST` (rest if ≥ 0.7) and
  `p[:,1]` is `DIFFUSION` (unstable if > 0.3).
- **HOSTA:** predict `MAG` for all candidate sites at the current step, mask the current host, take the argmax.

## 6. Adapting to another MD system

Everything system-specific is hard-coded. Work through this table top to bottom.

| # | What is hard-coded | Where | What to change |
|---|---|---|---|
| 1 | 539 atoms per frame, 180 Ti, Ti first in the atom list | `raw_pos['TI'] = index % 539`, `TI_IONS`, `extract_s1a` (`i_539`, `^ 539`, `range(1,181)+[538,539]`), `dist[..., 539]` | Use your atom count and number of polaron-hosting cations. Better: read them from the OUTCAR (`NIONS`, ions per type). |
| 2 | Definition of the candidate sites | `extract_s1a`: `y_ref_1/2`, `z_ref`, tolerances | Replace with your own site selection (a different surface or layer, bulk, another cation). Keep the output schema `{site: {t1,t2,b1,b2,l,r,ti1,ti2}}`. `ase.neighborlist` or `pymatgen` are more robust than regex parsing. |
| 3 | Local coordination (6 O + 2 Ti, grouped by z) | `extract_s1a`, `O_1…O_6`, `Ti_1/2`, `AVG_*` | For another polyhedron or another number of neighbours, change the neighbour dictionary, the feature construction in both `generate_*` functions, and `feature_list` in both scripts. The network input size adapts automatically. |
| 4 | Reference bond lengths | `create_ref_dist_data.py` | Recompute from your relaxed (0 K) structure. |
| 5 | O-vacancy position (Å) | `o_vacancy_coord = (17.824, 9.86, 15.95)` (twice) | Set to your defect position or remove `DIST_VACANCY`. It is a constant per supercell. |
| 6 | Number of polarons | `find_polaron_paths` (`-2` in `argpartition`), `DIST_POL1/2` | Set `n` = number of excess electrons or polarons and generate `DIST_POL1…n`. |
| 7 | Polaron magnetisation thresholds | `find_EoI(…, 0.2, 0.05)`, `generate_clssification(…, 0.75)`, REST threshold `0.3`, 4-step stability window, 10-step skip | Look at `|MAG|` histograms of your system. Localised moment of the polaron cation and the vibrational noise level set the values. Also scale the windows to your MD time step. |
| 8 | Non-orthogonal cell | `distance_3d_periodic` (minimum image per axis) | Needs a proper minimum-image routine for triclinic cells. |
| 9 | File names and encodings | `outcars` list, path `if/elif` blocks, `utf-16` for `MD700K` | Generalise to one config dict per dataset and write the output name from it. |
| 10 | Different magnetic quantity | awk for magnetisation, `MAG` column | With Bader or per-site moments from another code, produce the same `MAG STEP` file. |
| 11 | Training data selection | `OUTCARFILE_ID*` in `train_models.py` | Point to your pickles and keep both scripts' `feature_list` consistent. |

After porting, check in this order: (a) the S1A / site list has the expected length, (b) `find_polaron_paths`
shows physically sensible hops (plot `TI` vs `TIME`), (c) the number of EoIs per Ti, (d) class balance of
`REST` vs `DIFFUSION`.

## 7. Quick start

```bash
mkdir -p data/labeled_data_v3
python label_events.py          # writes data/labeled_data_v3/*.pkl
python train_models.py          # HOSTA (ml_type = "REGRESSION")
# set ml_type = "CLASSIFICATION" and re-run for INSTA
```


---
## License

DynTrap is distributed under the Mozilla Public License 2.0 (MPL-2.0).
See [LICENSE](LICENSE) for the complete terms.

The MPL-2.0 permits use, modification, and redistribution, including commercial use.
If MPL-covered files are modified and distributed, the covered source files must
remain available under the MPL-2.0. Other software, including VASP, remains subject
to its own licensing terms. DynTrap does not provide a VASP license.

## Citation

If you use DynTrap in research or build on this code, **please cite**:

S. Trivisonne, C. Verdi, and M. Reticcioli,
“Machine-Learning Dynamics of Redox-Active Materials: Polaron Diffusion on the TiO2(110) Surface,”
*ACS Nano* (2026). https://doi.org/10.1021/acsnano.6c13435

The repository also includes a [`CITATION.cff`](CITATION.cff) file for GitHub's
**Cite this repository** feature.