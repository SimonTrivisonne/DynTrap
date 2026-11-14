# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
DynTrap main script: detect a polaron jump in a finished MD run and prepare the restart.

Pipeline (see ``main`` at the bottom of this file):

    1. Parse the OUTCAR of the finished MD run: ion positions of every ionic step, lattice vector
       lengths and the nearest-neighbour table (-> ``extract_s1a``, ``load_raw_positions``,
       ``read_lattice_vectors``).
    2. Build time-resolved structural features for every S1A Ti site (-> ``compute_input_features``).
    3. Predict the stability (classification) and magnetisation (regression) of every site and step with
       the trained neural networks, find the first polaron jump ("event of interest", EoI) and choose the
       Ti site the polaron jumps to (-> ``compute_predictions``).
    4. Write a log entry and a new POSCAR for the next MD segment (-> ``create_log``, ``create_POSCAR``).
    5. Print the step at which the next segment starts, so the calling job script can pick it up.

Index conventions
    * OUTCAR / POSCAR / log output use 1-based ion indices.
    * Inside this script all indices are 0-based (``extract_s1a`` output is shifted once in ``main``).
    * ``Time`` / ``STEP`` are 1-based ionic step numbers.
"""

from __future__ import annotations

import re
import os
import datetime

import pandas as pd
import numpy as np

from sklearn.preprocessing import PowerTransformer
from MDAnalysis.analysis.distances import distance_array
from keras.models import load_model


# ======================================================================================================
# Configuration
# ======================================================================================================
# Everything that is specific to one simulated system or to one trained model lives in this section.
# To apply DynTrap to another MD system, this is the place to start.

# --- Files -------------------------------------------------------------------------------------------
OUTCAR_FILE = "OUTCAR"
POSCAR_FILE = "POSCAR"
CONTCAR_FILE = "CONTCAR"
XDATCAR_FILE = "XDATCAR"
JOB_SCRIPT_FILE = "job.sh"
LOG_FILE = "polaron_dynamic_log.txt"
INSTA_MODEL_FILE = "INSTA.h5"
HOSTA_MODEL_FILE = "HOSTA.h5"

# --- Structure ---------------------------------------------------------------------------------------
N_IONS = 539                           # total number of ions in the cell (must match the OUTCAR)
POLARONIC_TI_1BASED = (538, 539)       # the two Ti ions that carry the polarons (OUTCAR numbering)
S1A_CANDIDATES_1BASED = list(range(1, 181)) + list(POLARONIC_TI_1BASED)   # Ti ions checked for "S1A"

# Fractional-coordinate criteria that identify an S1A Ti site (see ``_select_s1a_ions``)
S1A_Y_REFERENCES = (0.999, 0.498)
S1A_Y_WRAP_THRESHOLD = 0.95            # |y - y_ref_1| above this means "close to y_ref_1 across the periodic edge"
S1A_Z_REFERENCE = 0.364
S1A_COORDINATE_TOLERANCE = 0.05

# Tolerances used while classifying the neighbours of an S1A Ti (fractional coordinates)
SAME_LEVEL_Z_TOLERANCE = 0.02          # |dz| <= tol: neighbour is on the same level as the Ti (left/right)
TI_NEIGHBOUR_X_TOLERANCE = 0.15        # used to find Ti neighbours missing from the OUTCAR table
TI_NEIGHBOUR_Y_TOLERANCE = 0.01

# Oxygen vacancy position (Cartesian, Angstrom)
O_VACANCY_COORD = np.array([17.824, 9.86, 15.95])

# Number of lines of a POSCAR/CONTCAR: 9 header lines, positions, one blank line, velocities
POSCAR_HEADER_LINES = 9
CONTCAR_N_LINES = POSCAR_HEADER_LINES + N_IONS + 1 + N_IONS

# --- Detection of the jump and choice of the target site ---------------------------------------------
STABILITY_THRESHOLD = 0.75             # class-1 probability from which a site counts as "unstable" (class 1)
N_IGNORED_STEPS = 20                   # the first steps of every trajectory are ignored
STABLE_WINDOW_LABEL_SPAN = 10          # a jump needs class 1 on the rows i ... i+10 (label slice, inclusive)
VACANCY_GAUSSIAN_SIGMA = 8             # width of the vacancy weighting (Angstrom)
POL2_LINEAR_SCALE = 12                 # linear weight towards the 2nd polaron: distance / scale - offset
POL2_LINEAR_OFFSET = 1 / 4
ROW_SEPARATION_Y_RANGE = (5, 8)        # |dy| (Angstrom) between two sites that sit in neighbouring S1A rows
RANDOM_JUMP_THRESHOLD = 0.02           # all weighted magnetisations below this -> random ("artificial") jump

NO_EVENT = -1                          # marker: "no polaron jump detected"

# --- Features ----------------------------------------------------------------------------------------
# Order matters: the neural networks expect the features in exactly this order.
BOND_FEATURES = ['Ti_1', 'Ti_2', 'O_1', 'O_2', 'O_3', 'O_4', 'O_5', 'O_6']
AVERAGED_FEATURES = ['AVG_Ti', 'AVG_O_T', 'AVG_O_B', 'AVG_O_AP']
POLARON_DISTANCE_FEATURES = ['DIST_POL1', 'DIST_POL2']

# Features for which first (d...) and second (dd...) time derivatives are computed
TIME_SERIES_FEATURES = BOND_FEATURES + AVERAGED_FEATURES + POLARON_DISTANCE_FEATURES
FIRST_DERIVATIVE_FEATURES = ['d' + name for name in TIME_SERIES_FEATURES]
SECOND_DERIVATIVE_FEATURES = ['dd' + name for name in TIME_SERIES_FEATURES]

FEATURES_CLASSIFICATION = TIME_SERIES_FEATURES + FIRST_DERIVATIVE_FEATURES + SECOND_DERIVATIVE_FEATURES
FEATURES_REGRESSION = (BOND_FEATURES + AVERAGED_FEATURES + ['DIST_VACANCY'] + POLARON_DISTANCE_FEATURES
                       + FIRST_DERIVATIVE_FEATURES)

# (feature name, key in the neighbour dict of ``extract_s1a``, key in the reference-distance dict)
# O_1/O_2 = top oxygens, O_3/O_4 = bottom oxygens, O_5/O_6 = left/right ("ap") oxygens
BOND_DEFINITIONS = [
    ('Ti_1', 'ti1', 'ti'), ('Ti_2', 'ti2', 'ti'),
    ('O_1', 't1', 't'), ('O_2', 't2', 't'),
    ('O_3', 'b1', 'b'), ('O_4', 'b2', 'b'),
    ('O_5', 'l', 'ap'), ('O_6', 'r', 'ap'),
]
# (feature name, first bond feature, second bond feature)
AVERAGE_DEFINITIONS = [
    ('AVG_Ti', 'Ti_1', 'Ti_2'),
    ('AVG_O_T', 'O_1', 'O_2'),
    ('AVG_O_B', 'O_3', 'O_4'),
    ('AVG_O_AP', 'O_5', 'O_6'),
]
# Neighbour keys that are averaged into one reference distance (see ``_compute_reference_distances``)
REFERENCE_DISTANCE_GROUPS = {
    'ti': ('ti1', 'ti2'),
    't': ('t1', 't2'),
    'b': ('b1', 'b2'),
    'ap': ('r', 'l'),
}

# --- OUTCAR / POSCAR parsing -------------------------------------------------------------------------
NEIGHBOUR_TABLE_HEADER = ' ion  position               nearest neighbor table\n'
WHOLE_NUMBER = re.compile(r'(?<!\.)\b\d+\b(?!\.)')       # integers, but not parts of decimal numbers
FRACTIONAL_COORDINATE = re.compile(r'(?:0|1)\.\d+')      # decimal numbers starting with "0." or "1."

# awk programs, executed as ``awk '<program>' <file>`` by ``_run_awk``
# One line per ion and ionic step: "x y z STEP" (taken from the "POSITION ... TOTAL-FORCE" blocks)
AWK_ION_POSITIONS = "BEGIN{STEP=1} {if ($1 ~ /POSITION/) {getline; getline; while (!($1 ~ /--/)) { print $1, $2, $3, STEP; getline;}; STEP++}}"
# Three lines: the lengths of the lattice vectors
AWK_LATTICE_LENGTHS = "/length of vectors/ {getline; print $1; print $2; print $3; exit}"
# Three lines: number of ions per type (Ti type 1, O, Ti type 2)
AWK_IONS_PER_TYPE = "/ions per type/ {print $5; print $6; print $7; exit}"
# One line per ion: the three selective-dynamics flags (T/F) that follow the position in the POSCAR
AWK_POSCAR_FLAGS = "/Direct/ {flag=1; next} flag && NF {print $4, $5, $6} !NF {flag=0; exit}"
# The header (first lines) of the POSCAR
AWK_POSCAR_HEADER = f"NR <= {POSCAR_HEADER_LINES}"


# ======================================================================================================
# Small helpers
# ======================================================================================================

def _run_awk(program: str, filename: str) -> list[str]:
    """Run ``awk '<program>' <filename>`` and return the output lines (without trailing newlines)."""
    output = os.popen(f"awk '{program}' {filename}").read()
    return output.split("\n")[:-1]


def _swap(items: list, pos1: int, pos2: int) -> None:
    """Swap two entries of a list in place."""
    items[pos1], items[pos2] = items[pos2], items[pos1]


def _in_neighbouring_row(y_a: float, y_b: float) -> bool:
    """True if two y coordinates belong to neighbouring S1A rows (distance within ``ROW_SEPARATION_Y_RANGE``)."""
    low, high = ROW_SEPARATION_Y_RANGE
    return low < abs(y_a - y_b) < high


# ======================================================================================================
# Step 1: parse the OUTCAR
# ======================================================================================================

def extract_s1a(file):
    """
    Find all S1A Ti sites in the OUTCAR and classify their neighbours.

    The OUTCAR contains a nearest-neighbour table (below the header ' ion  position   nearest neighbor
    table') that lists, for every ion, its fractional position followed by its neighbours sorted by
    distance. Because of that sorting the oxygen neighbours cannot be read off directly: we also need to
    know whether an oxygen is a top, bottom, left or right neighbour. For this the z component of the
    Ti is compared with the z components of its neighbours.

    Returns:
        dict {ti_index: {'r', 'l', 't1', 't2', 'b1', 'b2', 'ti1', 'ti2'}} with 1-based OUTCAR indices.
        r/l = oxygens on the same level, t1/t2 = oxygens above, b1/b2 = oxygens below, ti1/ti2 = Ti
        neighbours. A value of 0 means "not found".
    """
    with open(file, 'r') as outcar:
        lines = outcar.readlines()

    table_start = lines.index(NEIGHBOUR_TABLE_HEADER)
    table_end = _find_last_table_row(lines, table_start)
    table = lines[table_start:table_end + 1]

    x_components, y_components, z_components = _parse_fractional_positions(table)
    s1a_ions = _select_s1a_ions(y_components, z_components)
    neighbours = _classify_neighbours(table, s1a_ions, z_components)

    # Some Ti neighbours are not listed in the OUTCAR table; find them via their x/y components.
    _fill_missing_ti_neighbour(neighbours, x_components, y_components, missing_key='ti2', other_key='ti1')
    _fill_missing_ti_neighbour(neighbours, x_components, y_components, missing_key='ti1', other_key='ti2')
    return neighbours


def _find_last_table_row(lines: list[str], table_start: int) -> int:
    """Line index of the neighbour-table row of the last ion (index ``N_IONS``)."""
    for offset, line in enumerate(lines[table_start:]):
        if re.search(f"^ {N_IONS}", line):
            return table_start + offset
    raise ValueError(f"Neighbour table row of ion {N_IONS} not found in the OUTCAR")


def _parse_fractional_positions(table: list[str]):
    """Read the fractional x, y and z coordinate of every ion from the neighbour table.

    Returns three dicts {ion_index: coordinate}."""
    x_components, y_components, z_components = {}, {}, {}
    for line in table:
        # ignore lines without a whole number (empty lines)
        match = WHOLE_NUMBER.search(line)
        if match is None:
            continue
        # the ion index is the first whole number in the row
        ion = int(match.group())
        # the first three decimal numbers in the row are the x, y and z component
        components = FRACTIONAL_COORDINATE.findall(line)
        if not components:
            continue
        x_components[ion] = float(components[0])
        y_components[ion] = float(components[1])
        z_components[ion] = float(components[2])
    return x_components, y_components, z_components


def _select_s1a_ions(y_components: dict, z_components: dict) -> list[int]:
    """Select the S1A Ti ions: y close to ~0.999 (also across the periodic edge) or ~0.498, and z ~0.364."""
    y_ref_1, y_ref_2 = S1A_Y_REFERENCES
    tolerance = S1A_COORDINATE_TOLERANCE
    s1a_ions = []
    for ion in S1A_CANDIDATES_1BASED:
        y_diff_1 = abs(y_components[ion] - y_ref_1)
        y_diff_2 = abs(y_components[ion] - y_ref_2)
        z_diff = abs(z_components[ion] - S1A_Z_REFERENCE)
        y_matches = y_diff_1 < tolerance or y_diff_1 > S1A_Y_WRAP_THRESHOLD or y_diff_2 < tolerance
        if y_matches and z_diff < tolerance:
            s1a_ions.append(ion)
    return s1a_ions


def _classify_neighbours(table: list[str], s1a_ions: list[int], z_components: dict) -> dict:
    """Build the neighbour dict of every S1A Ti by comparing z components (see ``extract_s1a``)."""
    neighbours = {}
    for line in table:
        if WHOLE_NUMBER.search(line) is None:
            continue
        # whole numbers in the row: [ion, 6 oxygen neighbours, (Ti neighbour 1, Ti neighbour 2)]
        numbers = list(map(int, WHOLE_NUMBER.findall(line)))
        ion = numbers[0]
        if ion not in s1a_ions:
            continue

        # Ti neighbours that are not in the table are initialised with 0 and completed later
        neighbours[ion] = {"r": 0, "l": 0,
                           "t1": 0, "t2": 0,
                           "b1": 0, "b2": 0,
                           "ti1": numbers[7] if len(numbers) >= 8 else 0,
                           "ti2": numbers[8] if len(numbers) >= 9 else 0}

        # classify each oxygen neighbour as same level (r/l), above (t1/t2) or below (b1/b2)
        for oxygen in numbers[1:7]:
            delta_z = z_components[oxygen] - z_components[ion]
            if abs(delta_z) <= SAME_LEVEL_Z_TOLERANCE:
                first_slot, second_slot = 'r', 'l'
            elif delta_z > 0:
                first_slot, second_slot = 't1', 't2'
            else:
                first_slot, second_slot = 'b1', 'b2'
            slot = first_slot if neighbours[ion][first_slot] == 0 else second_slot
            neighbours[ion][slot] = oxygen
    return neighbours


def _fill_missing_ti_neighbour(neighbours: dict, x_components: dict, y_components: dict,
                               missing_key: str, other_key: str) -> None:
    """Fill neighbour entries ``missing_key`` that are still 0 (in place).

    The neighbour is searched among the S1A Ti sites: it must not be the Ti itself or its ``other_key``
    neighbour and has to lie within the x/y tolerances."""
    for ion, entry in neighbours.items():
        if entry[missing_key] != 0:
            continue
        for candidate in neighbours.keys():
            if (candidate != entry[other_key] and candidate != ion
                    and abs(x_components[candidate] - x_components[ion]) < TI_NEIGHBOUR_X_TOLERANCE
                    and abs(y_components[candidate] - y_components[ion]) < TI_NEIGHBOUR_Y_TOLERANCE):
                entry[missing_key] = candidate
                break


def load_raw_positions() -> pd.DataFrame:
    """Positions of all ions in all ionic steps (OUTCAR order).

    Returns a DataFrame with columns X, Y, Z, STEP (1-based) and TI (0-based ion index)."""
    rows = _run_awk(AWK_ION_POSITIONS, OUTCAR_FILE)
    raw_pos = pd.DataFrame([[float(value) for value in row.split(" ")] for row in rows],
                           columns=["X", "Y", "Z", "STEP"])
    raw_pos['TI'] = list(raw_pos.index % N_IONS)
    return raw_pos


def read_lattice_vectors() -> tuple:
    """Lengths of the three lattice vectors (a, b, c) from the OUTCAR."""
    vectors = _run_awk(AWK_LATTICE_LENGTHS, OUTCAR_FILE)
    return float(vectors[0]), float(vectors[1]), float(vectors[2])


# ======================================================================================================
# Step 2: input features
# ======================================================================================================

def compute_input_features(ti_s1a, ti_polaronic, raw_pos, lattice_vec):
    """
    Compute the time-resolved input features of all S1A Ti sites.

    Args:
        ti_s1a: neighbour dict from ``extract_s1a`` (0-based indices).
        ti_polaronic: 0-based indices of the two polaronic Ti.
        raw_pos: DataFrame from ``load_raw_positions``.
        lattice_vec: lengths of the lattice vectors.

    Returns:
        DataFrame with one row per (Ti, Time) and one column per feature. The last three steps of every
        Ti are dropped because the second derivative is not defined there.
    """
    total_ions = _read_total_ion_count()
    dist_data = _compute_distance_matrices(raw_pos, lattice_vec, total_ions)
    ref_dist = _compute_reference_distances(dist_data, ti_s1a, ti_polaronic)
    return _generate_input_features(dist_data, ref_dist, ti_s1a, ti_polaronic, raw_pos, lattice_vec)


def _read_total_ion_count() -> int:
    """Total number of ions (Ti type 1 + O + Ti type 2) from the OUTCAR."""
    ions = [int(count) for count in _run_awk(AWK_IONS_PER_TYPE, OUTCAR_FILE)]
    ti_ions_type1, o_ions, ti_ions_type2 = ions[0], ions[1], ions[2]
    total_ti_ions = ti_ions_type1 + ti_ions_type2
    return o_ions + total_ti_ions


def _compute_distance_matrices(raw_pos: pd.DataFrame, lattice_vec, total_ions: int) -> np.ndarray:
    """Periodic ion-ion distance matrix for every ionic step.

    Returns an array of shape (n_steps, total_ions, total_ions)."""
    ion_steps = int(max(raw_pos["STEP"]))
    data = np.empty([ion_steps, total_ions, total_ions], dtype=float)
    grouped = raw_pos.groupby(raw_pos.STEP)
    for step in range(1, ion_steps + 1):
        pos_df = grouped.get_group(step)[["X", "Y", "Z"]]
        data[step - 1] = distance_array(pos_df.to_numpy(), pos_df.to_numpy(), box=list(lattice_vec) + [90, 90, 90])
    return data


def _compute_reference_distances(dist: np.ndarray, ti_s1a: dict, ti_polaronic) -> dict:
    """Mean bond lengths of all non-polaronic S1A Ti, used as reference for the bond features.

    Returns {'ti': Ti-Ti, 't': Ti-O(top), 'b': Ti-O(bottom), 'ap': Ti-O(left/right)}, averaged over
    all non-polaronic S1A Ti and all ionic steps."""
    collected = {group: [] for group in REFERENCE_DISTANCE_GROUPS}
    for ti, neighbours in ti_s1a.items():
        if ti in ti_polaronic:
            continue
        for group, neighbour_keys in REFERENCE_DISTANCE_GROUPS.items():
            for key in neighbour_keys:
                collected[group].append(dist[:, ti, neighbours[key]])
    return {group: np.mean(distances) for group, distances in collected.items()}


def _periodic_distance(reference_xyz, points_df: pd.DataFrame, box_size) -> np.ndarray:
    """
    Distance with periodic boundary conditions (minimum-image convention) between points.

    Args:
        reference_xyz: array-like with the coordinates of one point (shape (3,)) or of one point per
            row of ``points_df`` (shape (n, 3)).
        points_df: DataFrame with columns ['X', 'Y', 'Z'].
        box_size: size of the periodic box (box_x, box_y, box_z).

    Returns:
        np.ndarray with one distance per row of ``points_df``.
    """
    reference_xyz = np.array(reference_xyz)
    points = points_df[['X', 'Y', 'Z']].values

    # coordinate differences, wrapped into the periodic box
    diff = points - reference_xyz
    diff -= np.round(diff / np.array(box_size)) * np.array(box_size)

    return np.linalg.norm(diff, axis=1)


def _build_ti_features(ti, neighbours, distances, ref_dist, ti_pos, pol1_pos, pol2_pos, lattice_vec) -> pd.DataFrame:
    """All features of a single Ti over time (one row per ionic step)."""
    n_steps = len(distances)
    ti_xyz = ti_pos[['X', 'Y', 'Z']].values

    features = pd.DataFrame()
    features["Ti"] = [ti] * n_steps
    features["Time"] = list(range(1, n_steps + 1))

    # bond lengths relative to the reference bond lengths
    for name, neighbour_key, ref_key in BOND_DEFINITIONS:
        features[name] = distances[:, ti, neighbours[neighbour_key]] - ref_dist[ref_key]

    # mean of the two bonds of each kind
    for name, first, second in AVERAGE_DEFINITIONS:
        features[name] = np.add(features[first], features[second]) * 0.5

    # distances to the oxygen vacancy and to both polarons
    features["DIST_VACANCY"] = _periodic_distance(O_VACANCY_COORD, ti_pos, lattice_vec)
    features["DIST_POL1"] = _periodic_distance(ti_xyz, pol1_pos, lattice_vec)
    features["DIST_POL2"] = _periodic_distance(ti_xyz, pol2_pos, lattice_vec)

    # first derivative; the last step has none and is dropped
    for name in TIME_SERIES_FEATURES:
        features['d' + name] = np.append(np.diff(features[name], n=1), 0)
    features.drop(features.tail(1).index, inplace=True)

    # second derivative; the last two steps have none and are dropped
    for name in TIME_SERIES_FEATURES:
        features['dd' + name] = np.append(np.diff(features[name], n=2), [0, 0])
    features.drop(features.tail(2).index, inplace=True)
    return features


def _generate_input_features(data_distance, ref_dist, ti_s1a, ti_polaronic, raw_pos, lattice_vec) -> pd.DataFrame:
    """Stack the feature tables of all S1A Ti into a single DataFrame."""
    pol1_pos = raw_pos[raw_pos.TI == ti_polaronic[0]]
    pol2_pos = raw_pos[raw_pos.TI == ti_polaronic[1]]
    per_ti_features = []
    for ti, neighbours in ti_s1a.items():
        ti_pos = raw_pos[raw_pos.TI == ti]
        per_ti_features.append(
            _build_ti_features(ti, neighbours, data_distance, ref_dist, ti_pos, pol1_pos, pol2_pos, lattice_vec))
    return pd.concat(per_ti_features, ignore_index=True)


# ======================================================================================================
# Step 3: predictions, event of interest and target site
# ======================================================================================================

def compute_predictions(input_df, feature_list_reg, feature_list_clas, ti_polaronic, raw_pos, lattice_vec):
    """
    Predict stability and magnetisation, then locate the first polaron jump.

    Note: ``input_df`` is extended in place with the columns PRED_STABILITY, PRED_STABILITY_CERTAINTY and
    PRED_MAGNETIZATION.

    Returns:
        dict {'Ti_from', 'Ti_to', 'Time'} (0-based Ti indices). All values are ``NO_EVENT`` (-1) if no jump
        was detected.
    """
    _add_model_predictions(input_df, feature_list_reg, feature_list_clas, ti_polaronic)
    input_df_polaronic = input_df[(input_df.Ti == ti_polaronic[0]) | (input_df.Ti == ti_polaronic[1])]

    toi_ti_0 = _find_time_of_interest(input_df_polaronic, ti_polaronic[0])
    toi_ti_1 = _find_time_of_interest(input_df_polaronic, ti_polaronic[1])

    # use the polaron that becomes unstable first (on a tie: the second polaron)
    if toi_ti_0['time_from'] == NO_EVENT and toi_ti_1['time_from'] == NO_EVENT:
        return {'Ti_from': NO_EVENT, 'Time': NO_EVENT, 'Ti_to': NO_EVENT}
    elif toi_ti_0['time_from'] == NO_EVENT:
        first_toi = toi_ti_1
    elif toi_ti_1['time_from'] == NO_EVENT:
        first_toi = toi_ti_0
    elif toi_ti_0['time_from'] >= toi_ti_1['time_from']:
        first_toi = toi_ti_1
    else:
        first_toi = toi_ti_0
    return _select_target_site(first_toi, input_df, feature_list_clas, ti_polaronic, raw_pos, lattice_vec)


def _add_model_predictions(input_df, feature_list_reg, feature_list_clas, ti_polaronic) -> None:
    """Run both neural networks and add their predictions to ``input_df`` (in place)."""
    model_clas = load_model(INSTA_MODEL_FILE, compile=False)
    model_reg = load_model(HOSTA_MODEL_FILE, compile=False)
    input_df_polaronic = input_df[(input_df.Ti == ti_polaronic[0]) | (input_df.Ti == ti_polaronic[1])]

    # Normalise the data:
    #   regression     -> transformer fitted on the features of all S1A Ti
    #   classification -> transformer fitted on the features of the two polaronic Ti only
    scaler_reg = PowerTransformer()
    X_reg = scaler_reg.fit_transform(input_df[feature_list_reg])
    scaler_clas = PowerTransformer()
    scaler_clas.fit(input_df_polaronic[feature_list_clas])
    X_clas = scaler_clas.transform(input_df[feature_list_clas])

    pred_clas = model_clas.predict(X_clas, verbose=0)
    pred_mag = model_reg.predict(X_reg, verbose=0)

    # class 1 = unstable polaron; the certainty is the probability of the predicted class
    input_df['PRED_STABILITY'] = [1 if p[1] >= STABILITY_THRESHOLD else 0 for p in pred_clas]
    input_df['PRED_STABILITY_CERTAINTY'] = [pred_clas[i, p] for i, p in enumerate(input_df['PRED_STABILITY'])]
    input_df['PRED_MAGNETIZATION'] = pred_mag


def _find_time_of_interest(input_df_polaronic: pd.DataFrame, ti) -> dict:
    """
    Find the time of interest (ToI) of one polaronic Ti: the first period in which it is predicted unstable.

    The search looks for the first row ``i`` for which the rows ``i ... i+10`` (label slice, inclusive)
    all have PRED_STABILITY == 1. The first ``N_IGNORED_STEPS`` steps are ignored. The period ends at the
    first following row with PRED_STABILITY == 0 (it can be longer than the window).

    Returns:
        dict {'ti_from', 'time_from', 'time_to'}; time_from and time_to are ``NO_EVENT`` if no period was
        found (or if it does not end before the trajectory does).
    """
    x = input_df_polaronic[input_df_polaronic.Ti == ti][N_IGNORED_STEPS:]
    time_from = 0
    time_to = 0
    for i, row in x.iterrows():
        if all(x.loc[i:i + STABLE_WINDOW_LABEL_SPAN].PRED_STABILITY == 1):
            time_from = row.Time
            # search for the end of the unstable period
            for _, later_row in x.loc[i:].iterrows():
                if later_row.PRED_STABILITY == 0:
                    time_to = later_row.Time
                    break
            break
    if time_to == 0:
        return {'ti_from': ti, 'time_from': NO_EVENT, 'time_to': NO_EVENT}
    return {'ti_from': int(ti), 'time_from': int(time_from), 'time_to': int(time_to)}


def _periodic_axis_distances(points: np.ndarray, ref_points: np.ndarray, lattice_vec) -> np.ndarray:
    """Euclidean distance between corresponding rows of two (n, 3) arrays, with periodic boundaries."""
    dx = np.abs(points[:, 0] - ref_points[:, 0])
    dy = np.abs(points[:, 1] - ref_points[:, 1])
    dz = np.abs(points[:, 2] - ref_points[:, 2])

    # apply periodic boundary conditions
    dx = np.minimum(dx, lattice_vec[0] - dx)
    dy = np.minimum(dy, lattice_vec[1] - dy)
    dz = np.minimum(dz, lattice_vec[2] - dz)

    return np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)


def _gaussian_weights(points: np.ndarray, ref_points: np.ndarray, sigma: float, lattice_vec) -> np.ndarray:
    """Weights of points based on their distance to reference points: Gaussian with width ``sigma``."""
    distances = _periodic_axis_distances(points, ref_points, lattice_vec)
    return np.exp(-distances ** 2 / (2 * sigma ** 2))


def _linear_weights(points: np.ndarray, ref_points: np.ndarray, lattice_vec) -> np.ndarray:
    """Weights of points based on their distance to reference points: linear in the distance."""
    distances = _periodic_axis_distances(points, ref_points, lattice_vec)
    return (distances / POL2_LINEAR_SCALE) - POL2_LINEAR_OFFSET


def _coordinates_in_window(raw_pos: pd.DataFrame, ti, time_from: int, time_to: int) -> pd.DataFrame:
    """X, Y, Z of one Ti for the steps time_from ... time_to (inclusive)."""
    return raw_pos[(raw_pos.TI == ti) & (raw_pos.STEP.between(time_from, time_to))][['X', 'Y', 'Z']]


def _select_target_site(toi, input_df, feature_list_clas, ti_polaronic, raw_pos, lattice_vec) -> dict:
    """
    Find the most favourable Ti for the polaron that jumps during the time of interest ``toi``.

    Every candidate Ti (all S1A Ti except the two polaronic ones) gets
        WEIGHTED_MAG = |predicted magnetisation| * weight(vacancy distance) * weight(distance to 2nd polaron)
    and the site with the highest value at the time of highest prediction certainty is chosen. If all
    weighted magnetisations are tiny, the jump is "artificial" and the target is drawn randomly,
    proportional to the squared weights.

    Returns:
        dict {'Ti_from', 'Ti_to', 'Time'} (0-based Ti indices).
    """
    time_from, time_to, jumping_ti = toi['time_from'], toi['time_to'], toi['ti_from']
    other_polaron = ti_polaronic[0] if jumping_ti == ti_polaronic[1] else ti_polaronic[1]

    # candidate sites during the ToI; only the prediction columns are needed from here on
    sites = input_df[(input_df.Ti != ti_polaronic[0]) & (input_df.Ti != ti_polaronic[1])
                     & (input_df.Time.between(time_from, time_to))]
    sites = sites.drop(columns=feature_list_clas)

    # coordinates of the jumping polaron, the other polaron and the oxygen vacancy during the ToI
    pol1_coord = _coordinates_in_window(raw_pos, jumping_ti, time_from, time_to)
    pol2_coord = _coordinates_in_window(raw_pos, other_polaron, time_from, time_to)
    vacancy_coord = np.tile(O_VACANCY_COORD, (int(time_to - time_from + 1), 1))

    pol1_y = pol1_coord.Y.iloc[0]
    pol2_in_other_row = _in_neighbouring_row(pol1_y, pol2_coord.Y.iloc[0])

    for ti in sites.Ti.unique():
        ti_coord = _coordinates_in_window(raw_pos, ti, time_from, time_to)
        weights_pol2 = _linear_weights(ti_coord.to_numpy(), pol2_coord.to_numpy(), lattice_vec)
        weights_vacancy = _gaussian_weights(ti_coord.to_numpy(), vacancy_coord, VACANCY_GAUSSIAN_SIGMA, lattice_vec)
        is_current_ti = sites.Ti == ti

        # Weight with respect to the other polaron: a Ti gets the linear weight if it is in the same row
        # as the other polaron, otherwise a constant weight of 1. "Same row as the other polaron" is
        #   - same row as the jumping polaron,        if both polarons are in the same row
        #   - not in that row (a neighbouring row),   if the polarons are in different rows
        # which is equivalent to the comparison below.
        ti_in_other_row = _in_neighbouring_row(pol1_y, ti_coord.Y.iloc[0])
        if ti_in_other_row == pol2_in_other_row:
            sites.loc[is_current_ti, 'WEIGHTS_POL2'] = abs(weights_pol2)
        else:
            sites.loc[is_current_ti, 'WEIGHTS_POL2'] = 1
        sites.loc[is_current_ti, 'WEIGHTS_VACANCY'] = abs(weights_vacancy)
    sites['WEIGHTED_MAG'] = abs(sites.PRED_MAGNETIZATION) * sites.WEIGHTS_VACANCY * sites.WEIGHTS_POL2

    is_artificial_jump = all(sites.WEIGHTED_MAG < RANDOM_JUMP_THRESHOLD)

    # time within the ToI at which the jumping polaron is predicted unstable with the highest certainty
    jumping_ti_window = input_df[(input_df.Time.between(time_from, time_to)) & (input_df.Ti == jumping_ti)]
    time = input_df.loc[jumping_ti_window.PRED_STABILITY_CERTAINTY.idxmax()].Time

    sites_at_time = sites[sites.Time == time]
    if not is_artificial_jump:
        # site with the maximal weighted magnetisation at that time
        ti_to = sites.loc[sites_at_time.WEIGHTED_MAG.idxmax()].Ti
    else:
        total_weights = sites_at_time.WEIGHTS_VACANCY * sites_at_time.WEIGHTS_POL2
        ti_to = np.random.choice(a=sites_at_time.Ti, p=(total_weights ** 2) / sum(total_weights ** 2))
    return {'Ti_from': int(jumping_ti), 'Ti_to': int(ti_to), 'Time': int(time)}


# ======================================================================================================
# Step 4: log and restart files
# ======================================================================================================

def create_POSCAR(t, ti_from, ti_to):
    """
    Write the POSCAR for the next MD segment.

    * No event (t == -1): continue from the end of the previous run, i.e. use the CONTCAR.
    * Event at step t: take positions and velocities of step t + 1 from the XDATCAR and swap the entries of
      ``ti_from`` and ``ti_to`` (the polaron "jumps" by exchanging the two Ti).

    Args:
        t: step of the event (or -1).
        ti_from, ti_to: 0-based Ti indices.
    """
    if t == NO_EVENT:
        _write_poscar_from_contcar()
    else:
        _write_poscar_from_xdatcar(t, ti_from, ti_to)


def _write_poscar_from_contcar() -> None:
    """Continue at the last timestep: POSCAR = first ``CONTCAR_N_LINES`` lines of the CONTCAR."""
    with open(CONTCAR_FILE, 'r') as contcar:
        with open(POSCAR_FILE, 'w') as poscar:
            contcar_lines = contcar.readlines()[:CONTCAR_N_LINES]
            poscar.write("".join(contcar_lines))


def _write_poscar_from_xdatcar(t: int, ti_from: int, ti_to: int) -> None:
    """New POSCAR from the XDATCAR frame of step t + 1 with Ti ``ti_from`` and ``ti_to`` swapped."""
    # selective-dynamics flags and header of the current POSCAR (read before it is overwritten)
    flags = [line + "\n" for line in _run_awk(AWK_POSCAR_FLAGS, POSCAR_FILE)]
    header = [line + "\n" for line in _run_awk(AWK_POSCAR_HEADER, POSCAR_FILE)]

    with open(XDATCAR_FILE, 'r') as xdatcar:
        xdatcar_lines = xdatcar.readlines()

    frame_marker = f"Direct configuration={t + 1:>6}\n"
    positions_start = xdatcar_lines.index(frame_marker) + 1
    positions = [line[:-1] + "\t" + flag
                 for line, flag in zip(xdatcar_lines[positions_start:positions_start + N_IONS], flags)]
    _swap(positions, ti_from, ti_to)

    # the velocities follow the positions after one blank line
    velocities_start = positions_start + N_IONS + 1
    velocities = xdatcar_lines[velocities_start:velocities_start + N_IONS]
    _swap(velocities, ti_from, ti_to)

    with open(POSCAR_FILE, "w") as poscar:
        poscar.write("".join(header))
        poscar.write("".join(positions))
        poscar.write("\n")
        poscar.write("".join(velocities))


def create_log(EoI):
    """Append the result of this run (jump or no jump) to the log file. Ti numbers are logged 1-based."""
    timestamp = datetime.datetime.now().strftime('%m-%d %H:%M:%S')
    with open(LOG_FILE, 'a') as log:
        if EoI['Time'] == NO_EVENT:
            log.write(f"  {timestamp}  no polaron jump detected\n")
        else:
            log.write(f"  {timestamp}")
            log.write(f"  polaron jump from Ti{EoI['Ti_from'] + 1} to Ti{EoI['Ti_to'] + 1} at Time={EoI['Time']}\n")


# ======================================================================================================
# Main
# ======================================================================================================

def _to_zero_based(neighbours: dict) -> dict:
    """Convert all indices of the ``extract_s1a`` result from 1-based (OUTCAR) to 0-based.

    Note: entries that are still 0 ("not found") become -1."""
    return {ti - 1: {key: index - 1 for key, index in entry.items()} for ti, entry in neighbours.items()}


def _print_restart_step(EoI) -> None:
    """Print the step at which the next MD segment starts (read by the calling job script)."""
    if EoI['Time'] == NO_EVENT:
        # no jump occurred, so continue at the end of the last MD: read the number of MD steps from job.sh
        with open(JOB_SCRIPT_FILE) as job:
            job_script = job.read()
            timesteps_md = int(re.findall(r'(?<=timestepsMD=)\d+', job_script)[0])
            # NOTE: 1 is subtracted twice here (kept from the original implementation)
            print(timesteps_md - 2)
    else:
        print(int(EoI['Time']))


def main():
    pd.options.mode.chained_assignment = None    # suppress pandas warning

    ti_s1a = _to_zero_based(extract_s1a(file=OUTCAR_FILE))
    ti_polaronic = np.add(POLARONIC_TI_1BASED, -1)

    raw_pos = load_raw_positions()
    lattice_vec = read_lattice_vectors()

    input_df = compute_input_features(ti_s1a, ti_polaronic, raw_pos, lattice_vec)
    EoI = compute_predictions(input_df, FEATURES_REGRESSION, FEATURES_CLASSIFICATION, ti_polaronic, raw_pos, lattice_vec)
    create_log(EoI)
    create_POSCAR(EoI['Time'], EoI['Ti_from'], EoI['Ti_to'])
    _print_restart_step(EoI)


if __name__ == '__main__':
    main()