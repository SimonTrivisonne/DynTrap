# DynTrap: Polaron Trapping Workflow for VASP

This repository contains the scripts and environment specification for running the **DynTrap** workflow on top of a VASP molecular dynamics simulation.

DynTrap periodically analyzes the VASP trajectory to detect polaron jumps, updates the ionic configuration accordingly, and reconstructs a consistent final `OUTCAR` file.

---

## Contents

- `env.yml`  
  Conda environment specification for DynTrap. It defines the Python version and required scientific Python stack (NumPy, pandas, MDAnalysis, TensorFlow, scikit-learn, etc.).

- `job.sh`  
  Job script that:
  - Creates (if needed) and activates the `DynTrap_env` conda environment from `env.yml`.
  - Orchestrates repeated VASP runs up to a total number of MD steps.
  - After each VASP run, calls the DynTrap Python driver to:
    - Detect if/when a polaron jump occurs within the current MD window.
    - Update `POSCAR` to the configuration at the jump time (or final step if no jump happens).
  - Moves each finished `OUTCAR` into `OUTCARS/OUTCAR_<t>` where `<t>` is the accumulated MD time.
  - After the loop finishes, runs `create_OUTCAR.py` to build a single consistent final `OUTCAR_FINAL`.

- `dyntrap_main.py` (DynTrap driver)  
  Main DynTrap analysis script. It:
  - Parses VASP outputs (`OUTCAR`, `XDATCAR`, `POSCAR`) to extract ionic positions and lattice vectors.
  - Identifies relevant Ti sites and their local O environments.
  - Builds time-resolved feature vectors describing local Ti–O environments and distances to polarons/vacancies.
  - Uses pre-trained neural-network models to:
    - Classify polaron stability over time.
    - Predict local magnetizations.
  - Detects the **Event of Interest (EoI)**: a polaron jump, including
    - origin Ti site,
    - destination Ti site,
    - corresponding MD time step.
  - Writes a log entry to `polaron_dynamic_log.txt`.
  - Updates `POSCAR` to the configuration at the detected jump time, swapping the relevant Ti indices.
  - Prints the MD step at which to continue the outer loop:
    - If a jump is detected: prints the internal jump time.
    - If no jump is detected: prints the last MD step of the current VASP run.

- `create_OUTCAR.py`  
  Post-processing script that:
  - Reads all segmented `OUTCAR_*` files in the `OUTCARS/` directory.
  - Reads DynTrap’s event log (`polaron_dynamic_log.txt`) and the job parameters from `job.sh`.
  - Reassembles the segmented `OUTCAR_*` files into a single, consistent `OUTCAR_FINAL`:
    - Ensures ionic step numbering is continuous.
    - Applies Ti index swaps so that atom IDs in `OUTCAR_FINAL` match those in the **initial** `POSCAR`, even after polaron hops.
  - Renumbers all ionic steps in `OUTCAR_FINAL` to be strictly sequential.

---

## Setup

1. **Create the conda environment** (optional if you let `job.sh` handle it):

   ```bash
   conda env create -n DynTrap_env -f env.yml
   ```

2. **Activate the environment**:

   ```bash
   conda activate DynTrap_env
   ```

3. Ensure the following VASP input files are present and correctly configured in your working directory:
   - `INCAR`
   - `POSCAR`
   - `POTCAR`
   - `KPOINTS`

---

## Running the Workflow

1. Submit or execute the job script in your batch system (or directly on a shell):

   ```bash
   bash job.sh
   ```

2. The script will:
   - Run VASP for MD segments of length `timestepsMD` (set inside `job.sh` and inserted as `NSW` into `INCAR`).
   - After each segment:
     - Call `dyntrap_main.py` to analyze the trajectory and adjust `POSCAR`.
     - Move `OUTCAR` into `OUTCARS/OUTCAR_<t>` with `<t>` the cumulative time.
   - Continue until the total number of MD steps `timesteps` is reached.

3. At the end of the loop:
   - `create_OUTCAR.py` is executed automatically by `job.sh`.
   - The final reconstructed trajectory is written to `OUTCAR_FINAL`.

---

## Typical Outputs

- `OUTCARS/OUTCAR_<t>`: segmented `OUTCAR` files for each MD window.
- `polaron_dynamic_log.txt`: human-readable log of detected polaron jumps (time and sites).
- `OUTCAR_FINAL`: reconstructed global `OUTCAR` with:
  - continuous ionic step numbering,
  - consistent Ti atom indices across all segments.

---

## Notes

- The neural-network models used by DynTrap must be present in the working directory (e.g. `STABILITY_CLASSIFICATION.h5`, `MAGNETIZATION_REGRESSION.h5`).
- Adjust `timesteps` and `timestepsMD` in `job.sh` to tune:
  - total MD time,
  - resolution at which DynTrap checks for polaron jumps.