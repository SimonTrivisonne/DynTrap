# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.


import numpy as np
import pandas as pd
import tqdm
import matplotlib.pyplot as plt
from scipy import linalg

from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer, StandardScaler, MinMaxScaler
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow import keras
# import keras_tuner as kt
from keras.models import load_model


# instantiate model
def reference_model_regression(X, i):
    if i == 0:
        mlp = keras.models.Sequential([
            keras.layers.Dense(128, activation="relu", input_shape=(X.shape[-1],)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(48, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(36, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(96, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(82, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(24, activation="relu"),
            keras.layers.Dense(1, activation="linear")])
    elif i == 1:
        mlp = keras.models.Sequential([
            keras.layers.Dense(64, activation="relu", input_shape=(X.shape[-1],)),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(1, activation="linear")])
    elif i == 2:
        mlp = keras.models.Sequential([
            keras.layers.Dense(64, activation="relu", input_shape=(X.shape[-1],)),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(1, activation="linear")])
    mlp.compile(loss=tf.keras.losses.MeanAbsoluteError(),
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                metrics=[tf.keras.losses.MeanAbsoluteError()])

    return mlp


def reference_model_classification(X, classes):
    mlp = keras.models.Sequential([
        keras.layers.Dense(64, activation="relu", input_shape=(X.shape[-1],)),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.5),
        # keras.layers.Dense(64, activation="relu"),
        # keras.layers.Dropout(0.5),
        # keras.layers.Dense(36, activation="relu"),
        # keras.layers.Dropout(0.5),
        # keras.layers.Dense(96, activation="relu"),
        # keras.layers.Dropout(0.5),
        # keras.layers.Dense(82, activation="relu"),
        # keras.layers.Dropout(0.5),
        # keras.layers.Dense(24, activation="relu"),
        keras.layers.Dense(len(classes), activation="sigmoid")])

    # select loss fuction and optimization scheme
    mlp.compile(loss=tf.keras.losses.binary_crossentropy,
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])

    return mlp


def generate_class_weights(class_series, multi_class=True, one_hot_encoded=False):
    """
    Method to generate class weights given a set of multi-class or multi-label labels, both one-hot-encoded or not.
    Some examples of different formats of class_series and their outputs are:
      - generate_class_weights(['mango', 'lemon', 'banana', 'mango'], multi_class=True, one_hot_encoded=False)
      {'banana': 1.3333333333333333, 'lemon': 1.3333333333333333, 'mango': 0.6666666666666666}
      - generate_class_weights([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]], multi_class=True, one_hot_encoded=True)
      {0: 0.6666666666666666, 1: 1.3333333333333333, 2: 1.3333333333333333}
      - generate_class_weights([['mango', 'lemon'], ['mango'], ['lemon', 'banana'], ['lemon']], multi_class=False, one_hot_encoded=False)
      {'banana': 1.3333333333333333, 'lemon': 0.4444444444444444, 'mango': 0.6666666666666666}
      - generate_class_weights([[0, 1, 1], [0, 0, 1], [1, 1, 0], [0, 1, 0]], multi_class=False, one_hot_encoded=True)
      {0: 1.3333333333333333, 1: 0.4444444444444444, 2: 0.6666666666666666}
    The output is a dictionary in the format { class_label: class_weight }. In case the input is one hot encoded, the class_label would be index
    of appareance of the label when the dataset was processed.
    In multi_class this is np.unique(class_series) and in multi-label np.unique(np.concatenate(class_series)).
    Author: Angel Igareta (angel@igareta.com)
    """
    if multi_class:
        # If class is one hot encoded, transform to categorical labels to use compute_class_weight
        if one_hot_encoded:
            class_series = np.argmax(class_series, axis=1)

        # Compute class weights with sklearn method
        class_labels = np.unique(class_series)
        class_weights = compute_class_weight(class_weight='balanced', classes=class_labels, y=class_series)
        return dict(zip(class_labels, class_weights))
    else:
        # It is neccessary that the multi-label values are one-hot encoded
        mlb = None
        if not one_hot_encoded:
            mlb = MultiLabelBinarizer()
            class_series = mlb.fit_transform(class_series)

        n_samples = len(class_series)
        n_classes = len(class_series[0])

        # Count each class frequency
        class_count = [0] * n_classes
        for classes in class_series:
            for index in range(n_classes):
                if classes[index] != 0:
                    class_count[index] += 1

        # Compute class weights using balanced method
        class_weights = [n_samples / (n_classes * freq) if freq > 0 else 1 for freq in class_count]
        class_labels = range(len(class_weights)) if mlb is None else mlb.classes_
        return dict(zip(class_labels, class_weights))


def check_data_classification():
    t = pd.read_pickle("data/labeled_data_v3/data_OUTCAR_1_2017_clas.pkl")
    p = mlp.predict(scaler.transform(t[feature_list]))
    t['PRED_REST'] = p[:, 0]
    t['PRED_DIFF'] = p[:, 1]
    t['PRED_REST_s'] = [1 if r >= 0.7 else 0 for r in t.PRED_REST]
    t['PRED_DIFF_s'] = [1 if r > 0.3 else 0 for r in t.PRED_DIFF]
    pd.crosstab(t.REST, t.PRED_REST_s)


def check_data_regression(data, ti):
    data = data[data['TI'] == ti].sort_values('TIME').reset_index(drop=True)

    plt.subplot(2, 1, 1)
    plt.title("True Instability", fontsize=11)
    plt.plot(data.INSTABILITY)
    plt.ylabel("Ti_S1A", fontsize=8)

    plt.subplot(2, 1, 2)
    plt.title("MAG", fontsize=11)
    plt.plot(data.MAG)
    plt.ylabel("Ti_S1A", fontsize=8)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    """ ML model for site finding """
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    path = 'data/labeled_data_v3/'
    # ml_type = "CLASSIFICATION"
    ml_type = "REGRESSION"
    OUTCARFILE_ID = [0, 2, 3, 4, 5, 6]
    # OUTCARFILE_ID = [2]
    # OUTCARFILE_ID = [2, 3]
    OUTCARFILE_ID_2017 = range(1, 29)
    if ml_type == "REGRESSION":
        data = pd.concat([pd.read_pickle(path + f'data_OUTCAR_{id}_reg.pkl') for id in OUTCARFILE_ID] +
                         [pd.read_pickle(path + f'data_OUTCAR_{id}_2017_reg.pkl') for id in OUTCARFILE_ID_2017])
        feature_list = [
            'Ti_1', 'Ti_2', 'O_1', 'O_2', 'O_3', 'O_4', 'O_5', 'O_6',
            'AVG_Ti', 'AVG_O_T', 'AVG_O_B', 'AVG_O_AP',
            'Fx', 'Fy', 'Fz',
            'DIST_VACANCY', 'DIST_POL1', 'DIST_POL2',
            'dTi_1', 'dTi_2', 'dO_1', 'dO_2', 'dO_3', 'dO_4', 'dO_5', 'dO_6',
            'dAVG_Ti', 'dAVG_O_T', 'dAVG_O_B', 'dAVG_O_AP',
            'dDIST_POL1', 'dDIST_POL2',
        ]
    elif ml_type == "CLASSIFICATION":
        data = pd.concat(# [pd.read_pickle(path + f'data_OUTCAR_{id}_clas.pkl') for id in OUTCARFILE_ID] +
                         [pd.read_pickle(path + f'data_OUTCAR_{id}_2017_clas.pkl') for id in OUTCARFILE_ID_2017])
        # data = pd.concat([pd.read_pickle(path + f'data_OUTCAR_{id}_clas.pkl') for id in OUTCARFILE_ID])
        feature_list = [
            'Ti_1', 'Ti_2', 'O_1', 'O_2', 'O_3', 'O_4', 'O_5', 'O_6',
            'AVG_Ti', 'AVG_O_T', 'AVG_O_B', 'AVG_O_AP',
            'Fx', 'Fy', 'Fz',
            # 'DIST_POL1', 'DIST_POL2',
            'dTi_1', 'dTi_2', 'dO_1', 'dO_2', 'dO_3', 'dO_4', 'dO_5', 'dO_6',
            'dAVG_Ti', 'dAVG_O_T', 'dAVG_O_B', 'dAVG_O_AP',
            # 'dDIST_POL1', 'dDIST_POL2',
            'ddTi_1', 'ddTi_2', 'ddO_1', 'ddO_2', 'ddO_3', 'ddO_4', 'ddO_5', 'ddO_6',
            'ddAVG_Ti', 'ddAVG_O_T', 'ddAVG_O_B', 'ddAVG_O_AP',
            # 'ddDIST_POL1', 'ddDIST_POL2'
        ]


    classes = 'MAG' if ml_type == "REGRESSION" else ["REST", "DIFFUSION"]
    train_df, test_df = train_test_split(data, train_size=0.8, shuffle=False)

    X_train = train_df[feature_list]
    y_train = train_df[classes]

    X_test = test_df[feature_list]
    y_test = test_df[classes]
    print(f"train dp: {len(X_train)}\ntest dp: {len(X_test)}")
    if ml_type == "CLASSIFICATION":
        class_weights = generate_class_weights(y_train.values, True, True)
        print(class_weights)

    # TODO: look for other normalization method
    # feature scaling (standardization)
    scaler = PowerTransformer()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    #
    X_train = np.asarray(X_train).astype(np.float32)
    y_train = np.asarray(y_train).astype(np.float32)
    X_test = np.asarray(X_test).astype(np.float32)
    y_test = np.asarray(y_test).astype(np.float32)

    if ml_type == "REGRESSION":
        mlps = {}
        for i in range(3):
            mlps[i] = reference_model_regression(X_train, i)
            stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_mean_absolute_error', patience=10, restore_best_weights=True)
            history_regression = mlps[i].fit(X_train, y_train,
                                         batch_size=64, epochs=2000, verbose=1,
                                         validation_data=(X_test, y_test), shuffle=True,
                                         callbacks=[stop_early])
    elif ml_type == "CLASSIFICATION":
        mlp = reference_model_classification(X_train, classes)
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
        history_classification = mlp.fit(X_train, y_train,
                                         epochs=1000, verbose=1, batch_size=16,
                                         callbacks=[stop_early], validation_data=(X_test, y_test),
                                         shuffle=False, class_weight=class_weights)

    breakpoint()

    # mlp.save(path + f'mlp_model_2_OUTCAR_{"&".join(str(x) for x in OUTCARFILE_ID)}_{ml_type}.h5')