# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.


import copy
import re
import tqdm
import subprocess as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import chain
from keras.models import load_model

from create_ref_dist_data import REF_TI_DIST, REF_O_B_DIST, REF_O_T_DIST, REF_O_AP_DIST
from process_pos import proces_pos


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


def find_polaron_paths(mag, raw_pos):
    """ find the polaronic Ti atoms ids for each timestep, by finding the two largest Mag values of each timestep """
    mag = abs(mag)
    polaron_pos_df = pd.DataFrame(columns=['TIMESTEP', 'POLARON_ID', 'TI', 'X', 'Y', 'Z'])
    pol_ti_ids = []
    for t in range(len(mag)):
        pol_ti_ids.append(np.sort(np.argpartition(mag[t].flatten(), -2)[-2:]))

    for i, p in enumerate(pol_ti_ids[1:-2], start=1):
        if any(pol_ti_ids[i - 1] != p):
            if all(pol_ti_ids[i + 1] == pol_ti_ids[i - 1]):
                pol_ti_ids[i] = pol_ti_ids[i - 1]

    for t, p in enumerate(pol_ti_ids):
        raw_pos_1 = raw_pos[(raw_pos.TI == p[0]) & (raw_pos.STEP == t)]
        raw_pos_2 = raw_pos[(raw_pos.TI == p[1]) & (raw_pos.STEP == t)]
        polaron_pos_df = pd.concat([polaron_pos_df,
                                    pd.DataFrame({'TIMESTEP': [t], 'POLARON_ID': [1], 'TI': [p[0]],
                                                  'X': raw_pos_1.X.values,
                                                  'Y': raw_pos_1.Y.values,
                                                  'Z': raw_pos_1.Z.values}),
                                    pd.DataFrame({'TIMESTEP': [t], 'POLARON_ID': [2], 'TI': [p[1]],
                                                  'X': raw_pos_2.X.values,
                                                  'Y': raw_pos_2.Y.values,
                                                  'Z': raw_pos_2.Z.values}
                                                 )], ignore_index=True)
    return polaron_pos_df


def find_EoI(mag, ti_s1a, th_a, th_b):
    """ find alle timesteps where the mag drops from above th_a and stays below th_b, for each Ti """
    res = {}
    for ti in ti_s1a:
        ti_mag = np.abs(np.concatenate([x[ti] for x in mag]))
        ti_mag_th_a = np.where(ti_mag > th_a)[0]  # all indices where mag is above th_a
        if ti_mag_th_a.size == 0: continue
        index_of_intresst = np.where(np.diff(ti_mag_th_a) != 1)[0]  # all indices where the mag drops from above th_a
        EoI = []

        for i in index_of_intresst:
            for t in range(ti_mag_th_a[i], ti_mag_th_a[i + 1]):
                window = ti_mag[t: min(t + 4, len(ti_mag))]
                if np.all(window < th_b):
                    EoI.append(t)
                    break
        # this loop is for EoI that are at the end of simulation (mag stays above th_a)
        for t in range(ti_mag_th_a[-1], len(ti_mag)):
            if np.all(ti_mag[t:(t + 4 if t + 4 < len(ti_mag) else len(ti_mag))] < th_b):
                EoI.append(ti_mag_th_a[-1])
                break
        if not EoI: continue
        res[ti] = EoI
    return res


def distance_3d_periodic(point, points_df, box_size):
    """
    Compute the 3D distance with periodic boundary conditions between a point and all points in a pandas DataFrame.

    Parameters:
        point (tuple or list): The coordinates of the point in the format (x, y, z).
        points_df (pandas DataFrame): DataFrame containing coordinates of points with columns ['X', 'Y', 'Z'].
        box_size (tuple or list): The size of the periodic box in the format (box_x, box_y, box_z).

    Returns:
        pandas Series: Series containing distances between the given point and all points in points_df with periodic boundary conditions.
    """
    point = np.array(point)
    points_array = points_df[['X', 'Y', 'Z']].values

    # Calculate differences in coordinates with periodic boundary conditions
    diff = points_array - point
    diff -= np.round(diff / np.array(box_size)) * np.array(box_size)

    # Calculate distances
    distances = np.linalg.norm(diff, axis=1)
    return distances


def generate_clssification(EoI, mag, th):
    res = pd.DataFrame(columns=['TI', 'EoI_ID', 'TIME', 'MAG', 'REST', 'DIFFUSION'])
    for ti, EoI_list in EoI.items():
        for i, e in enumerate(EoI_list):
            if e == 0: continue
            ti_mag = np.abs(np.concatenate(mag[:e, ti]))
            tau = np.where(ti_mag > th)[0]  # get all indices before EoI to have mag > th
            if tau.size == 0: continue
            tau = tau[-1]  # get last index to have mag > th before EoI
            if any(ti_mag[tau:e] < 0.05): continue  # ignores data where the MAG spikes after an DIFF event
            if len(mag[tau:e, ti]) == 0: continue  # ignores data with MAG spkies
            ti_mag_tau = np.abs(np.concatenate(mag[tau:e, ti]))
            for t, (m, inst) in enumerate(zip(ti_mag_tau, np.linspace(0, 1, e - tau))):
                res = pd.concat([res, pd.DataFrame(
                    {'TI': [ti], 'EoI_ID': [i], 'TIME': [tau + t], 'MAG': [m], 'REST': [0], 'DIFFUSION': [1]})])

            if i == 0:
                if len(mag[:tau, ti]) == 0: continue
                # ti_mag_rest_index = np.where(np.abs(np.concatenate(mag[:tau, ti])) > 0.8)[0]
                ti_mag_rest_index = np.where(np.abs(np.concatenate(mag[:tau, ti]))>0.3)[0]
            else:
                if len(mag[EoI_list[i - 1]:tau, ti]) == 0: continue
                # ti_mag_rest_index = np.where(np.abs(np.concatenate(mag[EoI_list[i - 1]:tau, ti])) > 0.8)[0] + EoI_list[i - 1]
                ti_mag_rest_index = np.where(np.abs(np.concatenate(mag[EoI_list[i - 1]:tau, ti]))>0.3)[0] + EoI_list[i - 1]

            for j, (t, m) in enumerate(zip(ti_mag_rest_index, ti_mag[ti_mag_rest_index])):
                if j<10: continue
                # if j>400: break
                res = pd.concat([res, pd.DataFrame(
                    {'TI': [ti], 'EoI_ID': [-(i + 1)], 'TIME': [t], 'MAG': [m], 'REST': [1], 'DIFFUSION': [0]})])
    return res.reset_index(drop=True)


def generate_sitefinding_input(dist, pol_paths, mag, raw_pos, s1a, lattice_vec, max_amount_datapoints=2000):
    """ generate input for most favorable site finding ML model """
    input_df = pd.DataFrame(columns=['TI', 'TIME', 'EoI', 'MAG'])
    for ti in ti_s1a_atoms_indices.keys():
        input_df = pd.concat([input_df, pd.DataFrame({
            # 'TI': [ti] * len(dist), 'TIME': list(range(len(dist))),
            'TI': [ti] * len(dist), 'TIME': list(range(len(dist))),
            'EoI': [0] * len(dist), 'MAG': abs(mag[:, ti][:, 0])
        })])
    input_df.reset_index(drop=True, inplace=True)

    o_vacancy_coord = np.array((17.824, 9.86, 15.95))
    pol1_path = pol_paths[pol_paths.POLARON_ID == 1]
    pol2_path = pol_paths[pol_paths.POLARON_ID == 2]
    for ti in input_df['TI'].unique():
        ti_pos = raw_pos[raw_pos.TI == ti]
        # timesteps_of_event = input_df.loc[input_df['TI'] == ti, 'TIME'].to_list()
        input_df.loc[input_df.TI == ti, 'Ti_1'] = dist[:, ti, s1a[ti]['ti1']] - REF_TI_DIST
        input_df.loc[input_df.TI == ti, 'Ti_2'] = dist[:, ti, s1a[ti]['ti2']] - REF_TI_DIST
        input_df.loc[input_df.TI == ti, 'O_1'] = dist[:, ti, s1a[ti]['t1']] - REF_O_T_DIST
        input_df.loc[input_df.TI == ti, 'O_2'] = dist[:, ti, s1a[ti]['t2']] - REF_O_T_DIST
        input_df.loc[input_df.TI == ti, 'O_3'] = dist[:, ti, s1a[ti]['b1']] - REF_O_B_DIST
        input_df.loc[input_df.TI == ti, 'O_4'] = dist[:, ti, s1a[ti]['b2']] - REF_O_B_DIST
        input_df.loc[input_df.TI == ti, 'O_5'] = dist[:, ti, s1a[ti]['l']] - REF_O_AP_DIST
        input_df.loc[input_df.TI == ti, 'O_6'] = dist[:, ti, s1a[ti]['r']] - REF_O_AP_DIST

        input_df.loc[input_df['TI'] == ti, 'Fx'] = ti_pos["Fx"].values
        input_df.loc[input_df['TI'] == ti, 'Fy'] = ti_pos["Fy"].values
        input_df.loc[input_df['TI'] == ti, 'Fz'] = ti_pos["Fz"].values

        input_df.loc[input_df.TI == ti, 'AVG_Ti'] = np.mean([input_df.loc[input_df.TI == ti, 'Ti_1'],
                                                             input_df.loc[input_df.TI == ti, 'Ti_2']], axis=0)
        input_df.loc[input_df.TI == ti, 'AVG_O_T'] = np.mean([input_df.loc[input_df.TI == ti, 'O_1'],
                                                              input_df.loc[input_df.TI == ti, 'O_2']], axis=0)
        input_df.loc[input_df.TI == ti, 'AVG_O_B'] = np.mean([input_df.loc[input_df.TI == ti, 'O_3'],
                                                              input_df.loc[input_df.TI == ti, 'O_4']], axis=0)
        input_df.loc[input_df.TI == ti, 'AVG_O_AP'] = np.mean([input_df.loc[input_df.TI == ti, 'O_5'],
                                                               input_df.loc[input_df.TI == ti, 'O_6']], axis=0)

        input_df.loc[input_df.TI == ti, 'DIST_VACANCY'] = distance_3d_periodic(o_vacancy_coord, ti_pos, lattice_vec)
        input_df.loc[input_df.TI == ti, 'DIST_POL1'] = distance_3d_periodic(ti_pos[['X', 'Y', 'Z']].values, pol1_path,
                                                                            lattice_vec)
        input_df.loc[input_df.TI == ti, 'DIST_POL2'] = distance_3d_periodic(ti_pos[['X', 'Y', 'Z']].values, pol2_path,
                                                                            lattice_vec)

        # return input_df.reset_index(drop=True)
        for eoi_id in input_df.loc[input_df['TI'] == ti, 'EoI'].unique():
            input_df.loc[(input_df['TI'] == ti) & (input_df['EoI'] == eoi_id), 'dTi_1'] = np.diff(
                np.append(0, input_df.loc[(input_df['TI'] == ti) & (input_df['EoI'] == eoi_id), 'Ti_1']), n=1)
            input_df.loc[(input_df['TI'] == ti) & (input_df['EoI'] == eoi_id), 'dTi_2'] = np.diff(
                np.append(0, input_df.loc[(input_df['TI'] == ti) & (input_df['EoI'] == eoi_id), 'Ti_2']), n=1)
            input_df.loc[(input_df['TI'] == ti) & (input_df['EoI'] == eoi_id), 'dO_1'] = np.diff(
                np.append(0, input_df.loc[(input_df['TI'] == ti) & (input_df['EoI'] == eoi_id), 'O_1']), n=1)
            input_df.loc[(input_df['TI'] == ti) & (input_df['EoI'] == eoi_id), 'dO_2'] = np.diff(
                np.append(0, input_df.loc[(input_df['TI'] == ti) & (input_df['EoI'] == eoi_id), 'O_2']), n=1)
            input_df.loc[(input_df['TI'] == ti) & (input_df['EoI'] == eoi_id), 'dO_3'] = np.diff(
                np.append(0, input_df.loc[(input_df['TI'] == ti) & (input_df['EoI'] == eoi_id), 'O_3']), n=1)
            input_df.loc[(input_df['TI'] == ti) & (input_df['EoI'] == eoi_id), 'dO_4'] = np.diff(
                np.append(0, input_df.loc[(input_df['TI'] == ti) & (input_df['EoI'] == eoi_id), 'O_4']), n=1)
            input_df.loc[(input_df['TI'] == ti) & (input_df['EoI'] == eoi_id), 'dO_5'] = np.diff(
                np.append(0, input_df.loc[(input_df['TI'] == ti) & (input_df['EoI'] == eoi_id), 'O_5']), n=1)
            input_df.loc[(input_df['TI'] == ti) & (input_df['EoI'] == eoi_id), 'dO_6'] = np.diff(
                np.append(0, input_df.loc[(input_df['TI'] == ti) & (input_df['EoI'] == eoi_id), 'O_6']), n=1)

            input_df.loc[(input_df['TI'] == ti) & (input_df['EoI'] == eoi_id), 'dAVG_Ti'] = np.diff(
                np.append(0, input_df.loc[(input_df['TI'] == ti) & (input_df['EoI'] == eoi_id), 'AVG_Ti']), n=1)
            input_df.loc[(input_df['TI'] == ti) & (input_df['EoI'] == eoi_id), 'dAVG_O_T'] = np.diff(
                np.append(0, input_df.loc[(input_df['TI'] == ti) & (input_df['EoI'] == eoi_id), 'AVG_O_T']), n=1)
            input_df.loc[(input_df['TI'] == ti) & (input_df['EoI'] == eoi_id), 'dAVG_O_B'] = np.diff(
                np.append(0, input_df.loc[(input_df['TI'] == ti) & (input_df['EoI'] == eoi_id), 'AVG_O_B']), n=1)
            input_df.loc[(input_df['TI'] == ti) & (input_df['EoI'] == eoi_id), 'dAVG_O_AP'] = np.diff(
                np.append(0, input_df.loc[(input_df['TI'] == ti) & (input_df['EoI'] == eoi_id), 'AVG_O_AP']), n=1)

            input_df.loc[(input_df['TI'] == ti) & (input_df['EoI'] == eoi_id), 'dDIST_POL1'] = np.diff(
                np.append(0, input_df.loc[(input_df['TI'] == ti) & (input_df['EoI'] == eoi_id), 'DIST_POL1']), n=1)
            input_df.loc[(input_df['TI'] == ti) & (input_df['EoI'] == eoi_id), 'dDIST_POL2'] = np.diff(
                np.append(0, input_df.loc[(input_df['TI'] == ti) & (input_df['EoI'] == eoi_id), 'DIST_POL2']), n=1)

            input_df.drop(input_df.loc[(input_df['TI'] == ti) & (input_df['EoI'] == eoi_id)].head(1).index,
                          inplace=True)
            input_df.reset_index(drop=True, inplace=True)

        for eoi_id in input_df.loc[input_df['TI'] == ti, 'EoI'].unique():
            input_df.loc[(input_df['TI'] == ti) & (input_df['EoI'] == eoi_id), 'ddTi_1'] = np.diff(
                np.append([0, 0], input_df.loc[(input_df['TI'] == ti) & (input_df['EoI'] == eoi_id), 'Ti_1']), n=2)
            input_df.loc[(input_df['TI'] == ti) & (input_df['EoI'] == eoi_id), 'ddTi_2'] = np.diff(
                np.append([0, 0], input_df.loc[(input_df['TI'] == ti) & (input_df['EoI'] == eoi_id), 'Ti_2']), n=2)
            input_df.loc[(input_df['TI'] == ti) & (input_df['EoI'] == eoi_id), 'ddO_1'] = np.diff(
                np.append([0, 0], input_df.loc[(input_df['TI'] == ti) & (input_df['EoI'] == eoi_id), 'O_1']), n=2)
            input_df.loc[(input_df['TI'] == ti) & (input_df['EoI'] == eoi_id), 'ddO_2'] = np.diff(
                np.append([0, 0], input_df.loc[(input_df['TI'] == ti) & (input_df['EoI'] == eoi_id), 'O_2']), n=2)
            input_df.loc[(input_df['TI'] == ti) & (input_df['EoI'] == eoi_id), 'ddO_3'] = np.diff(
                np.append([0, 0], input_df.loc[(input_df['TI'] == ti) & (input_df['EoI'] == eoi_id), 'O_3']), n=2)
            input_df.loc[(input_df['TI'] == ti) & (input_df['EoI'] == eoi_id), 'ddO_4'] = np.diff(
                np.append([0, 0], input_df.loc[(input_df['TI'] == ti) & (input_df['EoI'] == eoi_id), 'O_4']), n=2)
            input_df.loc[(input_df['TI'] == ti) & (input_df['EoI'] == eoi_id), 'ddO_5'] = np.diff(
                np.append([0, 0], input_df.loc[(input_df['TI'] == ti) & (input_df['EoI'] == eoi_id), 'O_5']), n=2)
            input_df.loc[(input_df['TI'] == ti) & (input_df['EoI'] == eoi_id), 'ddO_6'] = np.diff(
                np.append([0, 0], input_df.loc[(input_df['TI'] == ti) & (input_df['EoI'] == eoi_id), 'O_6']), n=2)

            input_df.loc[(input_df['TI'] == ti) & (input_df['EoI'] == eoi_id), 'ddAVG_Ti'] = np.diff(
                np.append([0, 0], input_df.loc[(input_df['TI'] == ti) & (input_df['EoI'] == eoi_id), 'AVG_Ti']), n=2)
            input_df.loc[(input_df['TI'] == ti) & (input_df['EoI'] == eoi_id), 'ddAVG_O_T'] = np.diff(
                np.append([0, 0], input_df.loc[(input_df['TI'] == ti) & (input_df['EoI'] == eoi_id), 'AVG_O_T']), n=2)
            input_df.loc[(input_df['TI'] == ti) & (input_df['EoI'] == eoi_id), 'ddAVG_O_B'] = np.diff(
                np.append([0, 0], input_df.loc[(input_df['TI'] == ti) & (input_df['EoI'] == eoi_id), 'AVG_O_B']), n=2)
            input_df.loc[(input_df['TI'] == ti) & (input_df['EoI'] == eoi_id), 'ddAVG_O_AP'] = np.diff(
                np.append([0, 0], input_df.loc[(input_df['TI'] == ti) & (input_df['EoI'] == eoi_id), 'AVG_O_AP']), n=2)

            input_df.loc[(input_df['TI'] == ti) & (input_df['EoI'] == eoi_id), 'ddDIST_POL1'] = np.diff(
                np.append([0, 0], input_df.loc[(input_df['TI'] == ti) & (input_df['EoI'] == eoi_id), 'DIST_POL1']), n=2)
            input_df.loc[(input_df['TI'] == ti) & (input_df['EoI'] == eoi_id), 'ddDIST_POL2'] = np.diff(
                np.append([0, 0], input_df.loc[(input_df['TI'] == ti) & (input_df['EoI'] == eoi_id), 'DIST_POL2']), n=2)

            input_df.drop(input_df.loc[(input_df['TI'] == ti) & (input_df['EoI'] == eoi_id)].head(2).index,
                          inplace=True)
            input_df.reset_index(drop=True, inplace=True)
    return input_df.reset_index(drop=True)


def generate_stability_input(dist, pol_paths, mag, raw_pos, s1a, lattice_vec, data):
    """ generate input for most stability classification ML model """
    o_vacancy_coord = np.array((17.824, 9.86, 15.95))
    for ti in data['TI'].unique():
        ti_pos = raw_pos[raw_pos.TI == ti]
        ti_pos = ti_pos[ti_pos.STEP.isin(data[data.TI == ti].TIME)]
        pol1_path = pol_paths[(pol_paths.POLARON_ID == 1) & (pol_paths.TIMESTEP.isin(data[data.TI == ti].TIME))]
        pol2_path = pol_paths[(pol_paths.POLARON_ID == 2) & (pol_paths.TIMESTEP.isin(data[data.TI == ti].TIME))]
        timesteps_of_event = data.loc[data['TI'] == ti, 'TIME'].to_list()
        data.loc[data['TI'] == ti, 'Ti_0'] = dist[timesteps_of_event, ti, 539] # get distance of ti to its temperatur=0 position
        data.loc[data['TI'] == ti, 'Ti_1'] = dist[timesteps_of_event, ti, s1a[ti]['ti1']] - REF_TI_DIST
        data.loc[data['TI'] == ti, 'Ti_2'] = dist[timesteps_of_event, ti, s1a[ti]['ti2']] - REF_TI_DIST
        data.loc[data['TI'] == ti, 'O_1'] = dist[timesteps_of_event, ti, s1a[ti]['t1']] - REF_O_T_DIST
        data.loc[data['TI'] == ti, 'O_2'] = dist[timesteps_of_event, ti, s1a[ti]['t2']] - REF_O_T_DIST
        data.loc[data['TI'] == ti, 'O_3'] = dist[timesteps_of_event, ti, s1a[ti]['b1']] - REF_O_B_DIST
        data.loc[data['TI'] == ti, 'O_4'] = dist[timesteps_of_event, ti, s1a[ti]['b2']] - REF_O_B_DIST
        data.loc[data['TI'] == ti, 'O_5'] = dist[timesteps_of_event, ti, s1a[ti]['l']] - REF_O_AP_DIST
        data.loc[data['TI'] == ti, 'O_6'] = dist[timesteps_of_event, ti, s1a[ti]['r']] - REF_O_AP_DIST

        data.loc[data['TI'] == ti, 'Fx'] = ti_pos["Fx"].values
        data.loc[data['TI'] == ti, 'Fy'] = ti_pos["Fy"].values
        data.loc[data['TI'] == ti, 'Fz'] = ti_pos["Fz"].values

        data.loc[data['TI'] == ti, 'AVG_Ti'] = np.mean([data.loc[data['TI'] == ti, 'Ti_1'],
                                                        data.loc[data['TI'] == ti, 'Ti_2']], axis=0)
        data.loc[data['TI'] == ti, 'AVG_O_T'] = np.mean([data.loc[data['TI'] == ti, 'O_1'],
                                                         data.loc[data['TI'] == ti, 'O_2']], axis=0)
        data.loc[data['TI'] == ti, 'AVG_O_B'] = np.mean([data.loc[data['TI'] == ti, 'O_3'],
                                                         data.loc[data['TI'] == ti, 'O_4']], axis=0)
        data.loc[data['TI'] == ti, 'AVG_O_AP'] = np.mean([data.loc[data['TI'] == ti, 'O_5'],
                                                          data.loc[data['TI'] == ti, 'O_6']], axis=0)

        data.loc[data.TI == ti, 'DIST_VACANCY'] = distance_3d_periodic(o_vacancy_coord, ti_pos, lattice_vec)
        data.loc[data.TI == ti, 'DIST_POL1'] = distance_3d_periodic(ti_pos[['X', 'Y', 'Z']].values, pol1_path, lattice_vec)
        data.loc[data.TI == ti, 'DIST_POL2'] = distance_3d_periodic(ti_pos[['X', 'Y', 'Z']].values, pol2_path, lattice_vec)

        for eoi_id in data.loc[data['TI'] == ti, 'EoI_ID'].unique():
            data.loc[(data['TI'] == ti) & (data['EoI_ID'] == eoi_id), 'dTi_0'] = np.diff(
                np.append(0, data.loc[(data['TI'] == ti) & (data['EoI_ID'] == eoi_id), 'Ti_0']), n=1)
            data.loc[(data['TI'] == ti) & (data['EoI_ID'] == eoi_id), 'dTi_1'] = np.diff(
                np.append(0, data.loc[(data['TI'] == ti) & (data['EoI_ID'] == eoi_id), 'Ti_1']), n=1)
            data.loc[(data['TI'] == ti) & (data['EoI_ID'] == eoi_id), 'dTi_2'] = np.diff(
                np.append(0, data.loc[(data['TI'] == ti) & (data['EoI_ID'] == eoi_id), 'Ti_2']), n=1)
            data.loc[(data['TI'] == ti) & (data['EoI_ID'] == eoi_id), 'dO_1'] = np.diff(
                np.append(0, data.loc[(data['TI'] == ti) & (data['EoI_ID'] == eoi_id), 'O_1']), n=1)
            data.loc[(data['TI'] == ti) & (data['EoI_ID'] == eoi_id), 'dO_2'] = np.diff(
                np.append(0, data.loc[(data['TI'] == ti) & (data['EoI_ID'] == eoi_id), 'O_2']), n=1)
            data.loc[(data['TI'] == ti) & (data['EoI_ID'] == eoi_id), 'dO_3'] = np.diff(
                np.append(0, data.loc[(data['TI'] == ti) & (data['EoI_ID'] == eoi_id), 'O_3']), n=1)
            data.loc[(data['TI'] == ti) & (data['EoI_ID'] == eoi_id), 'dO_4'] = np.diff(
                np.append(0, data.loc[(data['TI'] == ti) & (data['EoI_ID'] == eoi_id), 'O_4']), n=1)
            data.loc[(data['TI'] == ti) & (data['EoI_ID'] == eoi_id), 'dO_5'] = np.diff(
                np.append(0, data.loc[(data['TI'] == ti) & (data['EoI_ID'] == eoi_id), 'O_5']), n=1)
            data.loc[(data['TI'] == ti) & (data['EoI_ID'] == eoi_id), 'dO_6'] = np.diff(
                np.append(0, data.loc[(data['TI'] == ti) & (data['EoI_ID'] == eoi_id), 'O_6']), n=1)

            data.loc[(data['TI'] == ti) & (data['EoI_ID'] == eoi_id), 'dAVG_Ti'] = np.diff(
                np.append(0, data.loc[(data['TI'] == ti) & (data['EoI_ID'] == eoi_id), 'AVG_Ti']), n=1)
            data.loc[(data['TI'] == ti) & (data['EoI_ID'] == eoi_id), 'dAVG_O_T'] = np.diff(
                np.append(0, data.loc[(data['TI'] == ti) & (data['EoI_ID'] == eoi_id), 'AVG_O_T']), n=1)
            data.loc[(data['TI'] == ti) & (data['EoI_ID'] == eoi_id), 'dAVG_O_B'] = np.diff(
                np.append(0, data.loc[(data['TI'] == ti) & (data['EoI_ID'] == eoi_id), 'AVG_O_B']), n=1)
            data.loc[(data['TI'] == ti) & (data['EoI_ID'] == eoi_id), 'dAVG_O_AP'] = np.diff(
                np.append(0, data.loc[(data['TI'] == ti) & (data['EoI_ID'] == eoi_id), 'AVG_O_AP']), n=1)

            data.loc[(data['TI'] == ti) & (data['EoI_ID'] == eoi_id), 'dDIST_POL1'] = np.diff(
                np.append(0, data.loc[(data['TI'] == ti) & (data['EoI_ID'] == eoi_id), 'DIST_POL1']), n=1)
            data.loc[(data['TI'] == ti) & (data['EoI_ID'] == eoi_id), 'dDIST_POL2'] = np.diff(
                np.append(0, data.loc[(data['TI'] == ti) & (data['EoI_ID'] == eoi_id), 'DIST_POL2']), n=1)

            data.drop(data.loc[(data['TI'] == ti) & (data['EoI_ID'] == eoi_id)].head(1).index, inplace=True)
            data.reset_index(drop=True, inplace=True)

        for eoi_id in data.loc[data['TI'] == ti, 'EoI_ID'].unique():
            data.loc[(data['TI'] == ti) & (data['EoI_ID'] == eoi_id), 'ddTi_1'] = np.diff(
                np.append([0, 0], data.loc[(data['TI'] == ti) & (data['EoI_ID'] == eoi_id), 'Ti_1']), n=2)
            data.loc[(data['TI'] == ti) & (data['EoI_ID'] == eoi_id), 'ddTi_2'] = np.diff(
                np.append([0, 0], data.loc[(data['TI'] == ti) & (data['EoI_ID'] == eoi_id), 'Ti_2']), n=2)
            data.loc[(data['TI'] == ti) & (data['EoI_ID'] == eoi_id), 'ddO_1'] = np.diff(
                np.append([0, 0], data.loc[(data['TI'] == ti) & (data['EoI_ID'] == eoi_id), 'O_1']), n=2)
            data.loc[(data['TI'] == ti) & (data['EoI_ID'] == eoi_id), 'ddO_2'] = np.diff(
                np.append([0, 0], data.loc[(data['TI'] == ti) & (data['EoI_ID'] == eoi_id), 'O_2']), n=2)
            data.loc[(data['TI'] == ti) & (data['EoI_ID'] == eoi_id), 'ddO_3'] = np.diff(
                np.append([0, 0], data.loc[(data['TI'] == ti) & (data['EoI_ID'] == eoi_id), 'O_3']), n=2)
            data.loc[(data['TI'] == ti) & (data['EoI_ID'] == eoi_id), 'ddO_4'] = np.diff(
                np.append([0, 0], data.loc[(data['TI'] == ti) & (data['EoI_ID'] == eoi_id), 'O_4']), n=2)
            data.loc[(data['TI'] == ti) & (data['EoI_ID'] == eoi_id), 'ddO_5'] = np.diff(
                np.append([0, 0], data.loc[(data['TI'] == ti) & (data['EoI_ID'] == eoi_id), 'O_5']), n=2)
            data.loc[(data['TI'] == ti) & (data['EoI_ID'] == eoi_id), 'ddO_6'] = np.diff(
                np.append([0, 0], data.loc[(data['TI'] == ti) & (data['EoI_ID'] == eoi_id), 'O_6']), n=2)

            data.loc[(data['TI'] == ti) & (data['EoI_ID'] == eoi_id), 'ddAVG_Ti'] = np.diff(
                np.append([0, 0], data.loc[(data['TI'] == ti) & (data['EoI_ID'] == eoi_id), 'AVG_Ti']), n=2)
            data.loc[(data['TI'] == ti) & (data['EoI_ID'] == eoi_id), 'ddAVG_O_T'] = np.diff(
                np.append([0, 0], data.loc[(data['TI'] == ti) & (data['EoI_ID'] == eoi_id), 'AVG_O_T']), n=2)
            data.loc[(data['TI'] == ti) & (data['EoI_ID'] == eoi_id), 'ddAVG_O_B'] = np.diff(
                np.append([0, 0], data.loc[(data['TI'] == ti) & (data['EoI_ID'] == eoi_id), 'AVG_O_B']), n=2)
            data.loc[(data['TI'] == ti) & (data['EoI_ID'] == eoi_id), 'ddAVG_O_AP'] = np.diff(
                np.append([0, 0], data.loc[(data['TI'] == ti) & (data['EoI_ID'] == eoi_id), 'AVG_O_AP']), n=2)

            data.loc[(data['TI'] == ti) & (data['EoI_ID'] == eoi_id), 'ddDIST_POL1'] = np.diff(
                np.append([0, 0], data.loc[(data['TI'] == ti) & (data['EoI_ID'] == eoi_id), 'DIST_POL1']), n=2)
            data.loc[(data['TI'] == ti) & (data['EoI_ID'] == eoi_id), 'ddDIST_POL2'] = np.diff(
                np.append([0, 0], data.loc[(data['TI'] == ti) & (data['EoI_ID'] == eoi_id), 'DIST_POL2']), n=2)

            data.drop(data.loc[(data['TI'] == ti) & (data['EoI_ID'] == eoi_id)].head(2).index, inplace=True)
            data.reset_index(drop=True, inplace=True)
    return data.reset_index(drop=True)


if __name__ == "__main__":
    """ label events for Site-finding Model """
    outcars =[ {'path': 'data/MD700K/', 'outcar_ids': [0, 2, 3, 4, 5, 6]},
               {'path': 'data/database_1VO-700K_2017/', 'outcar_ids': list(range(1, 29))},
               # {'path': 'data/OUTCARS4/', 'outcar_ids': list(range(1000, 2456))}
               # {'path': 'data/MD300K/', 'outcar_ids': [0]}
               ]
    feature_list = [
        'Ti_1', 'Ti_2', 'O_1', 'O_2', 'O_3', 'O_4', 'O_5', 'O_6',
        'Fx', 'Fy', 'Fz',
        'AVG_Ti', 'AVG_O_T', 'AVG_O_B', 'AVG_O_AP',
        'DIST_VACANCY', 'DIST_POL1', 'DIST_POL2',
        'dTi_1', 'dTi_2', 'dO_1', 'dO_2', 'dO_3', 'dO_4', 'dO_5', 'dO_6',
        'dAVG_Ti', 'dAVG_O_T', 'dAVG_O_B', 'dAVG_O_AP',
        'dDIST_POL1', 'dDIST_POL2',
        'ddTi_1', 'ddTi_2', 'ddO_1', 'ddO_2', 'ddO_3', 'ddO_4', 'ddO_5', 'ddO_6',
        'ddAVG_Ti', 'ddAVG_O_T', 'ddAVG_O_B', 'ddAVG_O_AP',
        'ddDIST_POL1', 'ddDIST_POL2'
    ]
    res = pd.DataFrame(columns=['TI', 'TIME', 'EoI', 'MAG'] + feature_list)
    for path_outcar_ids in outcars:
        for OUTCARFILE_ID in tqdm.tqdm(path_outcar_ids['outcar_ids']):
            path = path_outcar_ids['path']
            if path == 'data/MD700K/':
                outcar_path = path + f'OUTCAR.{OUTCARFILE_ID}{"-nodefail" if OUTCARFILE_ID >= 3 else ""}'
            elif path == 'data/MD300K/':
                outcar_path = path + f'OUTCAR.{OUTCARFILE_ID}'
            else:
                outcar_path = path + f'OUTCAR_{OUTCARFILE_ID}'
            ti_s1a = extract_s1a(outcar_path)
            ti_s1a_atoms_indices = {k - 1: {k2: v2 - 1 for k2, v2 in v.items()} for k, v in ti_s1a.items()}
            vectors = sp.Popen(['awk',
                                '/length of vectors/ {getline; print $1; print $2; print $3; exit}',
                                outcar_path],
                               stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE).stdout.read().split(b"\n")
            lattice_vec = (float(vectors[0]), float(vectors[1]), float(vectors[2]))
            raw_pos = sp.Popen(['awk',
                                'BEGIN{STEP=0} {if ($1 ~ /POSITION/) {getline; getline; while (!($1 ~ /--/)) {print $1, $2, $3, $4, $5, $6, STEP; getline;}; STEP++}}',
                                outcar_path],
                                stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE).stdout.read().split(b"\n")
            raw_pos = pd.DataFrame([list(map(float, r.decode().split(" "))) for r in raw_pos[:-1]], columns=["X", "Y", "Z", "Fx", "Fy", "Fz", "STEP"])
            # raw_pos = pd.read_csv(path + f'POSITIONS_OUTCAR_{OUTCARFILE_ID}.dat', sep=" ", names=["X", "Y", "Z", "STEP"],
            #                       encoding='utf-16' if path == 'data/MD700K/' else 'utf-8')
            raw_pos['TI'] = list(raw_pos.index % 539)
            # dist = np.load(path + f'dist_OUTCAR_{OUTCARFILE_ID}.npy')
            dist = proces_pos(outcar_path, raw_pos)

            ## unkomment this to create the mag numpy array from MAGNETIZATION_OUTCAR_2.dat
            # TI_IONS = 180
            # ION_STEPS = len(dist)
            # mag_raw = sp.Popen(['awk',
            #                     'BEGIN{STEP = 0} { if ($1 ~ /magnetization/){getline; getline; getline; getline; while (!($1 ~ /--/)) { print $5, STEP; getline;}STEP++}}',
            #                     outcar_path],
            #                    stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE).stdout.read().split(b"\n")
            # mag_raw = pd.DataFrame([list(map(float, r.decode().split(" "))) for r in mag_raw[:-1]], columns=["MAG", "STEP"])
            mag_raw = pd.read_csv(path + f'MAGNETIZATIONS_OUTCAR_{OUTCARFILE_ID}.dat', sep=" ", names=["MAG", "STEP"],
                                  encoding='utf-16' if path == 'data/MD700K/' else 'utf-8')
            # create empty array
            mag = np.empty([ION_STEPS, TI_IONS, 1], dtype=float)
            grouped = mag_raw.groupby("STEP")

            # extract and assign magnetization values per ion
            for step in range(ION_STEPS):
                mag_df = grouped.get_group(step)[["MAG"]]

                for ion in list(range(TI_IONS)):
                    magnetizations = []
                    magnetizations.append(mag_df.iloc[ion, 0])
                    mag[step][ion] = magnetizations
            # np.save(path + f'mag_OUTCAR_{OUTCARFILE_ID}.npy', mag)
            # mag = np.load(path + f'mag_OUTCAR_{OUTCARFILE_ID}.npy')
            polaron_paths = find_polaron_paths(mag, raw_pos)

            """ label data for HOSTA """
            input_data = generate_sitefinding_input(dist, polaron_paths, mag, raw_pos, ti_s1a_atoms_indices,
                                                    lattice_vec)
            # RANDOM HOP in OUTCAR_11 of 2017 data, from Ti92 (id=91) to Ti5 (id=4)
            valid_timesteps = polaron_paths[polaron_paths.TI.isin(ti_s1a_atoms_indices.keys())].TIMESTEP.values
            input_data_reg = input_data[input_data.TIME.isin(valid_timesteps)]

            """ label data for INSTA """
            EoI = find_EoI(mag, ti_s1a_atoms_indices, 0.2, 0.05)
            # label_data = generate_regression(EoI, mag, 0.85)
            label_data = generate_clssification(EoI, mag, 0.75)
            input_data_clas = generate_stability_input(dist, polaron_paths, mag, raw_pos, ti_s1a_atoms_indices, lattice_vec,
                                                  label_data)
            # breakpoint()
            # res = pd.concat([res, input_data])
            # breakpoint()
            if path == 'data/MD700K/':
                input_data_reg.to_pickle(f'data/labeled_data_v3/data_OUTCAR_{OUTCARFILE_ID}_reg.pkl')
                input_data_clas.to_pickle(f'data/labeled_data_v3/data_OUTCAR_{OUTCARFILE_ID}_clas.pkl')
            elif path == 'data/database_1VO-700K_2017/':
                input_data_reg.to_pickle(f'data/labeled_data_v3/data_OUTCAR_{OUTCARFILE_ID}_2017_reg.pkl')
                input_data_clas.to_pickle(f'data/labeled_data_v3/data_OUTCAR_{OUTCARFILE_ID}_2017_clas.pkl')
            elif path == 'data/OUTCARS4/':
                input_data_reg.to_pickle(f'data/labeled_data_v3/data_OUTCAR_{OUTCARFILE_ID}_4_reg.pkl')
                input_data_clas.to_pickle(f'data/labeled_data_v3/data_OUTCAR_{OUTCARFILE_ID}_4_clas.pkl')
