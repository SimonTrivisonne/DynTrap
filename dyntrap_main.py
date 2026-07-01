
import re
import os
import datetime

import pandas as pd
import numpy as np

from sklearn.preprocessing import PowerTransformer
from MDAnalysis.analysis.distances import distance_array
from keras.models import load_model


def extract_s1a(file):
    """ To extract the indices of the Ti atoms and their neighbours, we use the OUTCAR.
        There the positions and indices of each atom are written after ' *ion *position *nearest neighbor table'.
        Since the neighbours in the OUTCAR list are sorted by distance, one cannot directly extract the Oxygen neighbours,
        since we also need the inforamtion if it is an top, bottom, left or right neighbour. For that we compare the
        z component of the current Ti with the neighbours. """
    neighbours = {}
    with open(file, 'r') as outcar:
        o = outcar.readlines()
        i = o.index(' ion  position               nearest neighbor table\n')
        # find line index for the 539th Ti atom neighbouring list
        for j, line in enumerate(o[i:]):
            if re.search("^ 539", line):
                i_539 = i + j
                break
        z_components = {}
        x_components = {}
        y_components = {}
        # find all x, y and z componets of each atom for later comparison
        for line in o[i:i_539+1]:
            # ignore empty lines
            if re.search(r'(?<!\.)\b\d+\b(?!\.)', line) is None:
                continue
            # find Ti index [it's the first whole number in the row]
            n = int(re.search(r'(?<!\.)\b\d+\b(?!\.)', line).group())
            # find all positional components of Ti
            components = re.findall(r'((0|1)\.\d+)', line)
            # extract the z component [it's the third decimal number in the row]
            if not components:
                continue

            # extract the x component [it's the first decimal number in the row]
            x_components[n] = eval(components[0][0])
            # extract the y component [it's the second decimal number in the row]
            y_components[n] = eval(components[1][0])
            # extract the z component [it's the third decimal number in the row]
            z_components[n] = eval(components[2][0])

        # find all Ti_S1A atoms: search for y values ~ 0.999 or 0.498 and z values ~ 0.364
        y_ref_1 = 0.999
        y_ref_2 = 0.498
        z_ref = 0.364
        ti_s1a = []
        for j in list(range(1, 181)) + [538, 539]:
            y_diff_1 = abs(y_components[j] - y_ref_1)
            y_diff_2 = abs(y_components[j] - y_ref_2)
            z_diff = abs(z_components[j] - z_ref)
            if (y_diff_1 < 0.05 or y_diff_1 > 0.95 or y_diff_2 < 0.05) and z_diff < 0.05:
                ti_s1a.append(j)

        # create Ti_S1A neighbours list by comparing the z components
        for line in o[i:i_539+1]:
            # ignore empty lines
            if re.search(r'(?<!\.)\b\d+\b(?!\.)', line) is None:
                continue
            # find all whole numbers in the row
            n = list(map(int, re.findall(r'(?<!\.)\b\d+\b(?!\.)', line)))
            # n = [int(a) for a in n]  # n[0] is the index of the Ti

            # only continue if the Ti is an S1A
            if n[0] not in ti_s1a:
                continue

            # get z component of current Ti
            z = z_components[n[0]]
            # initialise the neighbour list [some ti2 are not present in the OUTCAR, they will be added later]
            neighbours[n[0]] = {"r": 0, "l": 0,
                                "t1": 0, "t2": 0,
                                "b1": 0, "b2": 0,
                                "ti1": n[7] if len(n) >= 8 else 0, "ti2": n[8] if len(n) >= 9 else 0}

            # check weather the z component is lower, higher or on the same level and classify the neighbour
            for q in n[1:7]:
                delta_z = z_components[q] - z
                if abs(delta_z) <= 0.02:
                    direction = 'r' if neighbours[n[0]]['r'] == 0 else 'l'
                elif delta_z > 0:
                    direction = 't1' if neighbours[n[0]]['t1'] == 0 else 't2'
                else:
                    direction = 'b1' if neighbours[n[0]]['b1'] == 0 else 'b2'
                neighbours[n[0]][direction] = q

        # some Ti2 neighbours are not present in the OUTCAR, here we find them by using the x_component
        if any([n['ti2'] == 0 for n in neighbours.values()]):
            for k, n in neighbours.items():
                if n['ti2'] == 0:
                    for n2 in neighbours.keys():
                        if n2 != n['ti1'] and n2 != k and abs(x_components[n2] - x_components[k]) < 0.15 \
                                and abs(y_components[n2] - y_components[k]) < 0.01:
                            neighbours[k]['ti2'] = n2
                            break
        if any([n['ti1'] == 0 for n in neighbours.values()]):
            for k, n in neighbours.items():
                if n['ti1'] == 0:
                    for n1 in neighbours.keys():
                        if n1 != n['ti2'] and n1 != k and abs(x_components[n1] - x_components[k]) < 0.15 \
                                and abs(y_components[n1] - y_components[k]) < 0.01:
                            neighbours[k]['ti1'] = n1
                            break
    return neighbours


def compute_input_features(ti_s1a, ti_polaronic, raw_pos, lattice_vec):
    def compute_ref_dist(dist, ti_s1a, ti_polaronic):
        ti_dist = [];
        o_t_dist = [];
        o_b_dist = [];
        o_ap_dist = []
        for ti in ti_s1a.keys():
            if ti not in ti_polaronic:
                ti_dist.append(dist[:, ti, ti_s1a[ti]['ti1']])
                ti_dist.append(dist[:, ti, ti_s1a[ti]['ti2']])
                o_t_dist.append(dist[:, ti, ti_s1a[ti]['t1']])
                o_t_dist.append(dist[:, ti, ti_s1a[ti]['t2']])
                o_b_dist.append(dist[:, ti, ti_s1a[ti]['b1']])
                o_b_dist.append(dist[:, ti, ti_s1a[ti]['b2']])
                o_ap_dist.append(dist[:, ti, ti_s1a[ti]['r']])
                o_ap_dist.append(dist[:, ti, ti_s1a[ti]['l']])

        return {'ti': np.mean(ti_dist), 't': np.mean(o_t_dist), 'b': np.mean(o_b_dist), 'ap': np.mean(o_ap_dist)}

    def process_positions(data_position):
        ion_steps = int(max(data_position["STEP"]))
        data = np.empty([ion_steps, total_ions, total_ions], dtype=float)
        grouped = data_position.groupby(data_position.STEP)
        for step in range(1, ion_steps + 1):
            pos_df = grouped.get_group(step)[["X", "Y", "Z"]]
            data[step - 1] = distance_array(pos_df.to_numpy(), pos_df.to_numpy(), box=list(lattice_vec) + [90, 90, 90])
        return data

    def distance_3d_periodic(points_df1, points_df2, box_size):
        """
        Compute the 3D distance with periodic boundary conditions between a point and all points in a pandas DataFrame.

        Parameters:
            points_df1 (pandas DataFrame): DataFrame containing coordinates of points with columns ['X', 'Y', 'Z'].
            points_df2 (pandas DataFrame): DataFrame containing coordinates of points with columns ['X', 'Y', 'Z'].
            box_size (tuple or list): The size of the periodic box in the format (box_x, box_y, box_z).

        Returns:
            pandas Series: Series containing distances between the given point and all points in points_df with periodic boundary conditions.
        """
        points_df1 = np.array(points_df1)
        points_df2 = points_df2[['X', 'Y', 'Z']].values

        # Calculate differences in coordinates with periodic boundary conditions
        diff = points_df2 - points_df1
        diff -= np.round(diff / np.array(box_size)) * np.array(box_size)

        # Calculate distances
        distances = np.linalg.norm(diff, axis=1)
        return distances

    def generate_input_feat(data_distance):
        res_df = pd.DataFrame()
        o_vacancy_coord = np.array((17.824, 9.86, 15.95))
        pol1_pos = raw_pos[raw_pos.TI == ti_polaronic[0]]
        pol2_pos = raw_pos[raw_pos.TI == ti_polaronic[1]]
        for ti, neighbours in ti_s1a.items():
            ti_pos = raw_pos[raw_pos.TI == ti]
            temp_df = pd.DataFrame()
            temp_df["Ti"] = [ti] * len(data_distance)
            temp_df["Time"] = list(range(1, len(data_distance) + 1))

            temp_df["Ti_1"] = data_distance[:, ti, neighbours["ti1"]] - ref_dist["ti"]
            temp_df["Ti_2"] = data_distance[:, ti, neighbours["ti2"]] - ref_dist["ti"]
            temp_df["O_1"] = data_distance[:, ti, neighbours["t1"]] - ref_dist["t"]
            temp_df["O_2"] = data_distance[:, ti, neighbours["t2"]] - ref_dist["t"]
            temp_df["O_3"] = data_distance[:, ti, neighbours["b1"]] - ref_dist["b"]
            temp_df["O_4"] = data_distance[:, ti, neighbours["b2"]] - ref_dist["b"]
            temp_df["O_5"] = data_distance[:, ti, neighbours["l"]] - ref_dist["ap"]
            temp_df["O_6"] = data_distance[:, ti, neighbours["r"]] - ref_dist["ap"]

            temp_df["AVG_Ti"] = np.add(temp_df.Ti_1, temp_df.Ti_2) * 0.5
            temp_df["AVG_O_T"] = np.add(temp_df.O_1, temp_df.O_2) * 0.5
            temp_df["AVG_O_B"] = np.add(temp_df.O_3, temp_df.O_4) * 0.5
            temp_df["AVG_O_AP"] = np.add(temp_df.O_5, temp_df.O_6) * 0.5

            temp_df["DIST_VACANCY"] = distance_3d_periodic(o_vacancy_coord, ti_pos, lattice_vec)
            temp_df["DIST_POL1"] = distance_3d_periodic(ti_pos[['X', 'Y', 'Z']].values, pol1_pos, lattice_vec)
            temp_df["DIST_POL2"] = distance_3d_periodic(ti_pos[['X', 'Y', 'Z']].values, pol2_pos, lattice_vec)

            temp_df['dTi_1'] = np.append(np.diff(temp_df.Ti_1, n=1), 0)
            temp_df['dTi_2'] = np.append(np.diff(temp_df.Ti_2, n=1), 0)
            temp_df['dO_1'] = np.append(np.diff(temp_df.O_1, n=1), 0)
            temp_df['dO_2'] = np.append(np.diff(temp_df.O_2, n=1), 0)
            temp_df['dO_3'] = np.append(np.diff(temp_df.O_3, n=1), 0)
            temp_df['dO_4'] = np.append(np.diff(temp_df.O_4, n=1), 0)
            temp_df['dO_5'] = np.append(np.diff(temp_df.O_5, n=1), 0)
            temp_df['dO_6'] = np.append(np.diff(temp_df.O_6, n=1), 0)

            temp_df['dAVG_Ti'] = np.append(np.diff(temp_df.AVG_Ti, n=1), 0)
            temp_df['dAVG_O_T'] = np.append(np.diff(temp_df.AVG_O_T, n=1), 0)
            temp_df['dAVG_O_B'] = np.append(np.diff(temp_df.AVG_O_B, n=1), 0)
            temp_df['dAVG_O_AP'] = np.append(np.diff(temp_df.AVG_O_AP, n=1), 0)

            temp_df['dDIST_POL1'] = np.append(np.diff(temp_df.DIST_POL1, n=1), 0)
            temp_df['dDIST_POL2'] = np.append(np.diff(temp_df.DIST_POL2, n=1), 0)
            temp_df.drop(temp_df.tail(1).index, inplace=True)

            temp_df['ddTi_1'] = np.append(np.diff(temp_df.Ti_1, n=2), [0, 0])
            temp_df['ddTi_2'] = np.append(np.diff(temp_df.Ti_2, n=2), [0, 0])
            temp_df['ddO_1'] = np.append(np.diff(temp_df.O_1, n=2), [0, 0])
            temp_df['ddO_2'] = np.append(np.diff(temp_df.O_2, n=2), [0, 0])
            temp_df['ddO_3'] = np.append(np.diff(temp_df.O_3, n=2), [0, 0])
            temp_df['ddO_4'] = np.append(np.diff(temp_df.O_4, n=2), [0, 0])
            temp_df['ddO_5'] = np.append(np.diff(temp_df.O_5, n=2), [0, 0])
            temp_df['ddO_6'] = np.append(np.diff(temp_df.O_6, n=2), [0, 0])

            temp_df['ddAVG_Ti'] = np.append(np.diff(temp_df.AVG_Ti, n=2), [0, 0])
            temp_df['ddAVG_O_T'] = np.append(np.diff(temp_df.AVG_O_T, n=2), [0, 0])
            temp_df['ddAVG_O_B'] = np.append(np.diff(temp_df.AVG_O_B, n=2), [0, 0])
            temp_df['ddAVG_O_AP'] = np.append(np.diff(temp_df.AVG_O_AP, n=2), [0, 0])

            temp_df['ddDIST_POL1'] = np.append(np.diff(temp_df.DIST_POL1, n=2), [0, 0])
            temp_df['ddDIST_POL2'] = np.append(np.diff(temp_df.DIST_POL2, n=2), [0, 0])
            temp_df.drop(temp_df.tail(2).index, inplace=True)

            res_df = pd.concat([res_df, temp_df], ignore_index=True)
        return res_df

    ions = os.popen("""awk '/ions per type/ {print $5; print $6; print $7; exit}' OUTCAR""").read().split("\n")
    ions = [int(i) for i in ions[:-1]]
    ti_ions_type1 = ions[0]
    ti_ions_type2 = ions[2]
    total_ti_ions = ti_ions_type1 + ti_ions_type2
    o_ions = ions[1]
    total_ions = o_ions + total_ti_ions

    dist_data = process_positions(raw_pos)
    ref_dist = compute_ref_dist(dist_data, ti_s1a, ti_polaronic)

    input = generate_input_feat(dist_data)
    return input


def compute_predictions(input_df, feature_list_reg, feature_list_clas, ti_polaronic, raw_pos, lattice_vec):
    model_clas = load_model('STABILITY_CLASSIFICATION.h5', compile=False)
    model_reg = load_model('MAGNETIZATION_REGRESSION.h5', compile=False)
    input_df_polaronic = input_df[(input_df.Ti == ti_polaronic[0]) | (input_df.Ti == ti_polaronic[1])]
    ## Normalize data
    scaler = PowerTransformer()
    X_reg = scaler.fit_transform(input_df[feature_list_reg])
    scaler.fit_transform(input_df_polaronic[feature_list_clas])
    X_clas = scaler.transform(input_df[feature_list_clas])
    pred_clas = model_clas.predict(X_clas, verbose=0)
    pred_mag = model_reg.predict(X_reg, verbose=0)

    input_df['PRED_STABILITY'] = [1 if p[1] >= 0.75 else 0 for p in pred_clas]
    input_df['PRED_STABILITY_certanty'] = [pred_clas[i, p] for i, p in enumerate(input_df['PRED_STABILITY'])]
    input_df['PRED_MAGNETIZATION'] = pred_mag

    input_df_polaronic = input_df[(input_df.Ti == ti_polaronic[0]) | (input_df.Ti == ti_polaronic[1])]

    def find_ToI(ti):
        """ find Time of interest, by searching for the first 10 consecutive 1s in input_df_polaronic.PRED_STABILITY
            NOTE: the first 20 timesteps are ignored!! """
        x = input_df_polaronic[input_df_polaronic.Ti == ti][20:]
        time_from = 0
        time_to = 0
        for i, row in x.iterrows():
            if all(x.loc[i:i + 10].PRED_STABILITY == 1):
                time_from = row.Time
                # if we find 10 consecutive 1s, search for the end of the 1s (could be more than 10)
                for _, row2 in x.loc[i:].iterrows():
                    if row2.PRED_STABILITY == 0:
                        time_to = row2.Time
                        break
                break
        if time_to == 0:
            return {'ti_from': ti, 'time_from': -1, 'time_to': -1}
        return {'ti_from': int(ti), 'time_from': int(time_from), 'time_to': int(time_to)}

    def compute_weights_gaussian(points, ref_points, sigma):
        """
                Compute weights for points based on their distances to a reference point using a Gaussian function.

                Args:
                - points: Array-like object containing the points.
                - ref_point: Reference point.
                - sigma: Parameter controlling the spread of the Gaussian.

                Returns:
                - weights: Array of weights corresponding to each point.
                """

        dx = np.abs(points[:, 0] - ref_points[:, 0])  # x-coordinate distances
        dy = np.abs(points[:, 1] - ref_points[:, 1])  # y-coordinate distances
        dz = np.abs(points[:, 2] - ref_points[:, 2])  # z-coordinate distances

        # Apply periodic boundary conditions
        dx = np.minimum(dx, lattice_vec[0] - dx)
        dy = np.minimum(dy, lattice_vec[1] - dy)
        dz = np.minimum(dz, lattice_vec[2] - dz)

        distances = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)  # Euclidean distances
        weights = np.exp(-distances**2 / (2 * sigma ** 2))
        return weights

    def compute_weights_linear(points, ref_points):
        """
                Compute weights for points based on their distances to a reference point using a Linear function.

                Args:
                - points: Array-like object containing the points.
                - ref_point: Reference point.

                Returns:
                - weights: Array of weights corresponding to each point.
                """

        dx = np.abs(points[:, 0] - ref_points[:, 0])  # x-coordinate distances
        dy = np.abs(points[:, 1] - ref_points[:, 1])  # y-coordinate distances
        dz = np.abs(points[:, 2] - ref_points[:, 2])  # z-coordinate distances

        # Apply periodic boundary conditions
        dx = np.minimum(dx, lattice_vec[0] - dx)
        dy = np.minimum(dy, lattice_vec[1] - dy)
        dz = np.minimum(dz, lattice_vec[2] - dz)

        distances = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)  # Euclidean distances
        weights = (distances/12)-1/4
        return weights

    def find_site(ToI):
        """ find the most favorable site """
        o_vacancy_coord = np.array([[17.824, 9.86, 15.95]]*int(ToI['time_to']-ToI['time_from']+1))
        sites = input_df[(input_df.Ti != ti_polaronic[0]) & (input_df.Ti != ti_polaronic[1])
                         & (input_df.Time.between(ToI['time_from'], ToI['time_to']))]
        sites.drop(feature_list_clas, axis=1, inplace=True)
        # coordinates of jumping polaron
        pol1_coord = raw_pos[(raw_pos.TI == ToI['ti_from'])
                             & (raw_pos.STEP.between(ToI['time_from'], ToI['time_to']))][['X', 'Y', 'Z']]
        # coordinates of other polaron
        pol2_coord = raw_pos[(raw_pos.TI == (ti_polaronic[0] if ToI['ti_from'] == ti_polaronic[1] else ti_polaronic[1]))
                             & (raw_pos.STEP.between(ToI['time_from'], ToI['time_to']))][['X', 'Y', 'Z']]
        # check if polarons are in same S1A row
        if 5 < abs(pol1_coord.iloc[0].Y - pol2_coord.iloc[0].Y) < 8:
            pol_same_row = False
        else:
            pol_same_row = True

        for ti in sites.Ti.unique():
            # coordinates of the current ti of loop
            ti_coord = raw_pos[(raw_pos.TI == ti) &
                               (raw_pos.STEP.between(ToI['time_from'], ToI['time_to']))][['X', 'Y', 'Z']]
            weights_pol2 = compute_weights_linear(ti_coord.to_numpy(), pol2_coord.to_numpy())
            weights_vacancy = compute_weights_gaussian(ti_coord.to_numpy(), o_vacancy_coord, sigma=8)
            # two different scenarions are taken into account: polarons are in the same row or different rows
            if pol_same_row:
                if 5 < abs(ti_coord.iloc[0].Y - pol1_coord.iloc[0].Y) < 8:
                    # if the current ti is in a different row then the jumping polaron it gets a const weight
                    sites.loc[sites.Ti == ti, 'WEIGHTS_POL2'] = 1
                else:
                    # if the cuurent ti is in  the same row as the jumping polaron it gets a linear weight
                    sites.loc[sites.Ti == ti, 'WEIGHTS_POL2'] = abs(weights_pol2)
            elif not pol_same_row:
                if 5 < abs(ti_coord.iloc[0].Y - pol1_coord.iloc[0].Y) < 8:
                    # if the cuurent ti is in  the same row as the jumping polaron it gets a linear weight
                    sites.loc[sites.Ti == ti, 'WEIGHTS_POL2'] = abs(weights_pol2)
                else:
                    # if the current ti is in a different row then the jumping polaron it gets a const weight
                    sites.loc[sites.Ti == ti, 'WEIGHTS_POL2'] = 1
            sites.loc[sites.Ti == ti, 'WEIGHTS_VACANCY'] = abs(weights_vacancy)
        sites['WEIGHTED_MAG'] = abs(sites.PRED_MAGNETIZATION) * sites.WEIGHTS_VACANCY * sites.WEIGHTS_POL2

        # artifical random jump
        art_rand_jump = True if all(sites.WEIGHTED_MAG < 0.02) else False

        # get the time between time_from and time_to, with the max PRED_STABILITY_certanty
        time = input_df.loc[input_df[(input_df.Time.between(ToI['time_from'], ToI['time_to'])) &
                                     (input_df.Ti == ToI['ti_from'])].PRED_STABILITY_certanty.idxmax()].Time
        if not art_rand_jump:
            # get site with maximal weighted mag at timestep 'time'
            ti_to = sites.loc[sites[(sites.Time == time)].WEIGHTED_MAG.idxmax()].Ti
        else:
            sites['TOT_WEIGHTS'] = sites.WEIGHTS_VACANCY * sites.WEIGHTS_POL2
            a = sites[sites.Time == time]
            ti_to = np.random.choice(a=a.Ti, p=(a.TOT_WEIGHTS ** 2) / sum(a.TOT_WEIGHTS ** 2))
        return {'Ti_from': int(ToI['ti_from']), 'Ti_to': int(ti_to), 'Time': int(time)}

    ToI_ti_0 = find_ToI(ti_polaronic[0])
    ToI_ti_1 = find_ToI(ti_polaronic[1])

    if ToI_ti_0['time_from'] == -1 and ToI_ti_1['time_from'] == -1:
        return {'Ti_from': -1, 'Time': -1, 'Ti_to': -1}
    elif ToI_ti_0['time_from'] == -1:
        EoI = find_site(ToI_ti_1)
    elif ToI_ti_1['time_from'] == -1:
        EoI = find_site(ToI_ti_0)
    elif ToI_ti_0['time_from'] >= ToI_ti_1['time_from']:
        EoI = find_site(ToI_ti_1)
    else:
        EoI = find_site(ToI_ti_0)
    return EoI


def create_POSCAR(t, ti_from, ti_to):
    TF = os.popen("""awk '/Direct/ {flag=1; next} flag && NF {print $4, $5, $6} !NF {flag=0; exit}' POSCAR""").read().split("\n")
    TF = [r + "\n" for r in TF][:-1]
    header = os.popen("""awk 'NR <= 9' POSCAR""").read().split("\n")
    header = [r + "\n" for r in header][:-1]

    def swap(list, pos1, pos2):
        list[pos1], list[pos2] = list[pos2], list[pos1]
        return list

    if t == -1:
        ## if no EoI is found, we continue at the last timestep
        with open(f"CONTCAR", 'r') as contcar:
            with open(f"POSCAR", 'w') as poscar:
                c = contcar.readlines()[:1088]
                poscar.write("".join(c))
    else:
        with open(f"XDATCAR", 'r') as xdatcar:
            d = xdatcar.readlines()
            i = d.index(f"Direct configuration={' ' * (6 - len(str(t + 1)))}{t + 1}\n") + 1
            r_pos = [x[:-1] + "\t" + y for x, y in zip(d[i: i + 539], TF)]
            r_pos = swap(r_pos, ti_from, ti_to)
            r_vel = d[i + 540: i + 540 + 539]
            r_vel = swap(r_vel, ti_from, ti_to)
            with open("POSCAR", "w") as poscar:
                poscar.write("".join(header))
                poscar.write("".join(r_pos))
                poscar.write("\n")
                poscar.write("".join(r_vel))


def create_log(EoI):
    with open('polaron_dynamic_log.txt', 'a') as log:
        if EoI['Time'] == -1:
            log.write(f"  {datetime.datetime.now().strftime('%m-%d %H:%M:%S')}  no polaron jump detected\n")
        else:
            log.write(f"  {datetime.datetime.now().strftime('%m-%d %H:%M:%S')}")
            log.write(f"  polaron jump from Ti{EoI['Ti_from'] + 1} to Ti{EoI['Ti_to'] + 1} at Time={EoI['Time']}\n")


if __name__ == '__main__':
    pd.options.mode.chained_assignment = None    # suppress pandas warning
    ti_s1a = extract_s1a(file="OUTCAR")
    ti_s1a = {k1 - 1: {k2: v2 - 1 for k2, v2 in v1.items()} for k1, v1 in ti_s1a.items()}
    feature_list_clas = [
        'Ti_1', 'Ti_2', 'O_1', 'O_2', 'O_3', 'O_4', 'O_5', 'O_6',
        'AVG_Ti', 'AVG_O_T', 'AVG_O_B', 'AVG_O_AP',
        'DIST_POL1', 'DIST_POL2',
        'dTi_1', 'dTi_2', 'dO_1', 'dO_2', 'dO_3', 'dO_4', 'dO_5', 'dO_6',
        'dAVG_Ti', 'dAVG_O_T', 'dAVG_O_B', 'dAVG_O_AP',
        'dDIST_POL1', 'dDIST_POL2',
        'ddTi_1', 'ddTi_2', 'ddO_1', 'ddO_2', 'ddO_3', 'ddO_4', 'ddO_5', 'ddO_6',
        'ddAVG_Ti', 'ddAVG_O_T', 'ddAVG_O_B', 'ddAVG_O_AP',
        'ddDIST_POL1', 'ddDIST_POL2'
    ]
    feature_list_reg = [
        'Ti_1', 'Ti_2', 'O_1', 'O_2', 'O_3', 'O_4', 'O_5', 'O_6',
        'AVG_Ti', 'AVG_O_T', 'AVG_O_B', 'AVG_O_AP',
        'DIST_VACANCY', 'DIST_POL1', 'DIST_POL2',
        'dTi_1', 'dTi_2', 'dO_1', 'dO_2', 'dO_3', 'dO_4', 'dO_5', 'dO_6',
        'dAVG_Ti', 'dAVG_O_T', 'dAVG_O_B', 'dAVG_O_AP',
        'dDIST_POL1', 'dDIST_POL2',
    ]
    ti_polaronic = [538, 539]
    ti_polaronic = np.add(ti_polaronic, -1)

    raw_pos = os.popen("""awk 'BEGIN{STEP=1} {if ($1 ~ /POSITION/) {getline; getline; while (!($1 ~ /--/)) { print $1, $2, $3, STEP; getline;}; STEP++}}' OUTCAR""").read().split("\n")[:-1]
    raw_pos = pd.DataFrame([[float(x) for x in row.split(" ")] for row in raw_pos],
                             columns=["X", "Y", "Z", "STEP"])
    raw_pos['TI'] = list(raw_pos.index % 539)
    vectors = os.popen("""awk '/length of vectors/ {getline; print $1; print $2; print $3; exit}' OUTCAR""").read().split("\n")
    lattice_vec = (float(vectors[0]), float(vectors[1]), float(vectors[2]))

    input_df = compute_input_features(ti_s1a, ti_polaronic, raw_pos, lattice_vec)
    EoI = compute_predictions(input_df, feature_list_reg, feature_list_clas, ti_polaronic, raw_pos, lattice_vec)
    create_log(EoI)
    create_POSCAR(EoI['Time'], EoI['Ti_from'], EoI['Ti_to'])
    # if EoI['Time'] is -1, it means no jumping occurred, so continue at the end of the last FF-MD
    if EoI['Time'] == -1:
        with open('job.sh') as job:
            j = job.read()
            timestepsmd = int(re.findall(r'(?<=timestepsMD=)\d+', j)[0]) - 1
            print(timestepsmd - 1)
    else:
        print(int(EoI['Time']))
