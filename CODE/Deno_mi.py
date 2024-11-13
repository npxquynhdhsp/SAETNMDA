# %%
from xgboost import XGBClassifier
from sklearn.metrics import auc, roc_auc_score, accuracy_score, roc_curve, \
    precision_score, average_precision_score, precision_recall_curve
from scipy import interp
from tensorflow.keras.backend import l2_normalize, maximum
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as keras_layers
import tensorflow.keras.models as keras_models
import pickle
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

# %%
nloop = 1  # @Q ori 5
bgl = 1  # @Q ori 1
tile = 15  # ti le mau am, duong trong cac triplet, tile = -1 la lay het. Ko phai ti le am duong nhu bt.
ne = 100  # number of epoch of triplet, ori 100
alpha = 0.2 # margin
miemb_size, diemb_size = 128, 128
batch_size = 128
lrr = 0.2  # Q
xgne = 500  # XGboost, ori 500
nm, nd = 788, 374
di_num, mi_num = nd, nm
didim, midim = nd, nm
l_alpha = 0.75
fi_A = '../IN/Q18.3_HMDD3/mi_k/'
fi_feature = '../IN/Q18.3_HMDD3/mi_k/'
fi_out = './OUT Q_HMDD3/mi_k/Q18.3/'

# %%
def get_pair(ix):  # get pair indexes from y_4loai
    pair_trN = np.argwhere(full_pair4.values == 0)
    pair_trP = np.argwhere(full_pair4.values == 1)
    print('pair_trainP.shape', pair_trP.shape)
    print('pair_trainN.shape', pair_trN.shape)
    pair_trN = {'mir': pair_trN[:, 0], 'dis': pair_trN[:, 1]}
    pair_trP = {'mir': pair_trP[:, 0], 'dis': pair_trP[:, 1]}
    return pair_trP, pair_trN

def miNET():
    net = keras_models.Sequential()
    net.add(keras_layers.Dense(256))
    net.add(keras_layers.Dense(64))
    # net.add(keras_layers.Dense(args.miemb_size))
    # net.add(keras_layers.Lambda(lambda x: l2_normalize(x, axis=1)))
    return net

def diNET():
    net = keras_models.Sequential()
    net.add(keras_layers.Dense(256))
    net.add(keras_layers.Dense(64))
    # net.add(keras_layers.Dense(args.diemb_size))
    # net.add(keras_layers.Lambda(lambda x: l2_normalize(x, axis=1)))
    return net

def miRNA_auto_encoder(x_train):
    input = layers.Input(midim)
    encoded = layers.Dense(512, activation='elu')(input)
    decoded = layers.Dense(midim, activation='elu')(encoded)
    autoencoder = models.Model(inputs=input, outputs=decoded)
    encoder = models.Model(inputs=input, outputs=encoded)

    adam = Adam(learning_rate=0.001)
    autoencoder.compile(optimizer=adam, loss='mse')
    autoencoder.fit(x_train, x_train, epochs=500, batch_size=64, shuffle=True) # validation_split=0.2
    miRNA_encodeds = encoder.predict(x_train)
    return miRNA_encodeds

def miRNA_auto_encoder2(x_train):
    input = layers.Input(512)
    encoded = layers.Dense(256, activation='elu')(input)
    decoded = layers.Dense(512, activation='elu')(encoded)
    autoencoder = models.Model(inputs=input, outputs=decoded)
    encoder = models.Model(inputs=input, outputs=encoded)

    adam = Adam(learning_rate=0.001)
    autoencoder.compile(optimizer=adam, loss='mse')
    autoencoder.fit(x_train, x_train, epochs=500, batch_size=64, shuffle=True)
    miRNA_encodeds = encoder.predict(x_train)
    return miRNA_encodeds

def miRNA_auto_encoder3(x_train):
    input = layers.Input(256)
    encoded = layers.Dense(64, activation='elu')(input)
    decoded = layers.Dense(256, activation='elu')(encoded)
    autoencoder = models.Model(inputs=input, outputs=decoded)
    encoder = models.Model(inputs=input, outputs=encoded)

    adam = Adam(learning_rate=0.001)
    autoencoder.compile(optimizer=adam, loss='mse')
    autoencoder.fit(x_train, x_train, epochs=500, batch_size=64, shuffle=True)
    miRNA_encodeds = encoder.predict(x_train)
    return miRNA_encodeds

def dis_auto_encoder(x_train):
    input = layers.Input(didim)
    encoded = layers.Dense(512, activation='elu')(input)
    decoded = layers.Dense(didim, activation='elu')(encoded)
    autoencoder = models.Model(inputs=input, outputs=decoded)
    encoder = models.Model(inputs=input, outputs=encoded)

    adam = Adam(learning_rate=0.001)
    autoencoder.compile(optimizer=adam, loss='mse')
    autoencoder.fit(x_train, x_train, epochs=500, batch_size=64, shuffle=True)
    miRNA_encodeds = encoder.predict(x_train)
    return miRNA_encodeds

def dis_auto_encoder2(x_train):
    input = layers.Input(512)
    encoded = layers.Dense(256, activation='elu')(input)
    decoded = layers.Dense(512, activation='elu')(encoded)
    autoencoder = models.Model(inputs=input, outputs=decoded)
    encoder = models.Model(inputs=input, outputs=encoded)

    adam = Adam(learning_rate=0.001)
    autoencoder.compile(optimizer=adam, loss='mse')
    autoencoder.fit(x_train, x_train, epochs=500, batch_size=64, shuffle=True)
    miRNA_encodeds = encoder.predict(x_train)
    return miRNA_encodeds

def dis_auto_encoder3(x_train):
    input = layers.Input(256)
    encoded = layers.Dense(64, activation='elu')(input)
    decoded = layers.Dense(256, activation='elu')(encoded)
    autoencoder = models.Model(inputs=input, outputs=decoded)
    encoder = models.Model(inputs=input, outputs=encoded)

    adam = Adam(learning_rate=0.001)
    autoencoder.compile(optimizer=adam, loss='mse')
    autoencoder.fit(x_train, x_train, epochs=500, batch_size=64, shuffle=True)
    miRNA_encodeds = encoder.predict(x_train)
    return miRNA_encodeds

def triplet_loss1(y_true, y_pred, miemb_size = miemb_size, diemb_size = diemb_size):
    def loss_(y_true, y_pred):
        anchor = y_pred[:, :miemb_size]
        positive = y_pred[:, miemb_size:miemb_size + diemb_size]
        negative = y_pred[:, miemb_size + diemb_size:]

        positive_dist = tf.square(anchor - positive)
        negative_dist = tf.square(anchor - negative)

        l1 = tf.maximum(positive_dist - negative_dist + alpha, 0.)
        l2 = tf.maximum(0., alpha - positive_dist + negative_dist)  # Margin Ranking Loss
        return tf.reduce_mean(l1)*l_alpha + tf.reduce_mean(l2)*(1 -l_alpha)

    return loss_(y_true, y_pred)

def triplet_loss2(y_true, y_pred, miemb_size = miemb_size, diemb_size = diemb_size):
    def loss_(y_true, y_pred):
        anchor = y_pred[:, :diemb_size]
        positive = y_pred[:, diemb_size:diemb_size + miemb_size]
        negative = y_pred[:, diemb_size + miemb_size:]

        positive_dist = tf.square(anchor - positive)
        negative_dist = tf.square(anchor - negative)

        l1 = tf.maximum(positive_dist - negative_dist + alpha, 0.)
        l2 = tf.maximum(0., alpha - positive_dist + negative_dist)  # thu Margin Ranking Loss
        return tf.reduce_mean(l1) * l_alpha + tf.reduce_mean(l2) * (1 - l_alpha)
    return loss_(y_true, y_pred)


# %%
def train_triplet1(loop_i, ix):
    def gen_triplet_idx(loop_i, tile, ix):
        pair_trP, pair_trN = get_pair(ix)
        # np.random.seed(2022)
        np.random.seed()
        triplets = []
        # Scan all miRNAs
        for mir_i in range(mi_num):  # A.shape[0]
            dis_link_mir_i = pair_trP['dis'][pair_trP['mir'] == mir_i]
            dis_nolink_mir_i = pair_trN['dis'][pair_trN['mir'] == mir_i]
            for dis_link in dis_link_mir_i:
                np.random.shuffle(dis_nolink_mir_i)

                if (tile == -1) or (len(dis_nolink_mir_i) < tile):
                    tile = len(dis_nolink_mir_i)
                for dis_nolink in dis_nolink_mir_i[:tile]:
                    triplets.append((mir_i, dis_link, dis_nolink))

        pickle.dump({'triplets': triplets},  # save triplets
                    open(fi_out + "Triplet_sample_train/" + "L" + str(loop_i) + "_From_tripletnet1_mi" + str(ix) + ".pkl", "wb"))
        print('triplets.shape', len(triplets))
        return triplets

    def tripletNET_1():
        in1 = keras_layers.Input(midim)
        in2 = keras_layers.Input(didim)
        in3 = keras_layers.Input(didim)

        f_mi = miNET()
        f_d1 = diNET()
        f_d2 = diNET()

        mi = f_mi(in1)
        d1 = f_d1(in2)
        d2 = f_d1(in3)

        in1A = keras_layers.Input(64)
        in2A = keras_layers.Input(64)
        in3A = keras_layers.Input(64)

        miA, d1A, d2A = in1A, in2A, in3A
        mi = keras_layers.Concatenate()([mi, miA])
        d1 = keras_layers.Concatenate()([d1, d1A])
        d2 = keras_layers.Concatenate()([d2, d2A])

        out = keras_layers.Concatenate()([mi, d1, d2])

        final = keras_models.Model(inputs=[in1, in2, in3, in1A, in2A, in3A], outputs=out)
        return final

    def get_sample_triplet1(full_pair, train_or_test):  # train_or_test = [0, 1] (train) / [3, 4] (test)
        pair_N = np.argwhere(full_pair.values == train_or_test[0])
        pair_P = np.argwhere(full_pair.values == train_or_test[1])

        pair_N = {'mir': pair_N[:, 0], 'dis': pair_N[:, 1]}
        pair_P = {'mir': pair_P[:, 0], 'dis': pair_P[:, 1]}

        triplets, y = [], []
        for mir_i in range(mi_num):
            dis_link_mir_i = pair_P['dis'][pair_P['mir'] == mir_i]
            dis_nolink_mir_i = pair_N['dis'][pair_N['mir'] == mir_i]

            for dis_link in dis_link_mir_i:
                triplets.append((mir_i, dis_link, dis_link))
                y.append(1)

            for dis_nolink in dis_nolink_mir_i:
                triplets.append((mir_i, dis_nolink, dis_nolink))
                y.append(0)
        return triplets, y

    def xuly(triplets, loop_i, ix):
        miRNA_embeddings = miRNA_auto_encoder(misim_data)  # Dữ liệu miRNA
        miRNA_embeddings = miRNA_auto_encoder2(miRNA_embeddings)
        miRNA_embeddings = miRNA_auto_encoder3(miRNA_embeddings)

        disease_embeddings = dis_auto_encoder(disim_data)  # Dữ liệu disease
        disease_embeddings = dis_auto_encoder2(disease_embeddings)
        disease_embeddings = dis_auto_encoder3(disease_embeddings)

        # (1)
        idx = np.arange(len(triplets))
        # np.random.seed(2022)
        np.random.seed()
        np.random.shuffle(idx)

        triplets = np.array(triplets)
        tr_mi = misim_data.iloc[triplets[idx, 0]]
        tr_di1 = disim_data.iloc[triplets[idx, 1]]
        tr_di2 = disim_data.iloc[triplets[idx, 2]]
        print('tr_mi.shape, tr_di1.shape, tr_di2.shape', tr_mi.shape, tr_di1.shape, tr_di2.shape)

        tr_miA = miRNA_embeddings[triplets[idx, 0], :]
        tr_di1A = disease_embeddings[triplets[idx, 1], :]
        tr_di2A = disease_embeddings[triplets[idx, 2], :]

        # (2) #Train tripletNET_1
        print("\n--- Train tripletNET_1 ...")
        y = np.array([0] * len(triplets))  # mảng số 0, để khớp code
        tripletnet1 = tripletNET_1()
        tripletnet1.compile(loss=triplet_loss1, optimizer='adam')
        _ = tripletnet1.fit([tr_mi, tr_di1, tr_di2, tr_miA, tr_di1A, tr_di2A], y, epochs=ne, verbose=2)
        print("Done")

        # (3) #GET TEST TRIPLETNET + PREDICT NO USE XGBOOST
        te_triplets, te_y = get_sample_triplet1(full_pair4, [3, 4])
        te_triplets = np.array(te_triplets)
        te_mi = misim_data.iloc[te_triplets[:, 0]]
        te_di1 = disim_data.iloc[te_triplets[:, 1]]
        te_di2 = disim_data.iloc[te_triplets[:, 2]]

        te_miA = miRNA_embeddings[te_triplets[:, 0], :]
        te_di1A = disease_embeddings[te_triplets[:, 1], :]
        te_di2A = disease_embeddings[te_triplets[:, 2], :]

        te_distance = tripletnet1.predict([te_mi, te_di1, te_di2, te_miA, te_di1A, te_di2A])

        anchor = te_distance[:, :miemb_size]
        positive = te_distance[:, miemb_size:miemb_size + diemb_size]
        negative = te_distance[:, miemb_size + diemb_size:]

        # -- Save
        te_X = np.concatenate([anchor, positive], axis=1)
        pickle.dump([te_X, te_y], open(fi_out + "Data_test/" + "L" + str(loop_i) + "_From_tripletnet1_mi" + str(ix) + ".pkl", "wb"))

        # (4) #GET TRAIN SET FOR TRADITIONAL
        # print("\n\n--- Lay Train cho traditional, tripletnet1")
        tr_triplets, tr_y = get_sample_triplet1(full_pair4, [0, 1])
        # print(len(tr_triplets))
        # print(len(tr_y))

        idx = np.arange(len(tr_triplets))
        tr_triplets = np.array(tr_triplets)
        tr_mi = misim_data.iloc[tr_triplets[idx, 0]]
        tr_di1 = disim_data.iloc[tr_triplets[idx, 1]]
        tr_di2 = disim_data.iloc[tr_triplets[idx, 2]]

        tr_miA = miRNA_embeddings[tr_triplets[idx, 0], :]
        tr_di1A = disease_embeddings[tr_triplets[idx, 1], :]
        tr_di2A = disease_embeddings[tr_triplets[idx, 2], :]

        tr_distance = tripletnet1.predict([tr_mi, tr_di1, tr_di2, tr_miA, tr_di1A, tr_di2A])

        tr_anchor = tr_distance[:, :miemb_size]
        tr_positive = tr_distance[:, miemb_size:miemb_size + diemb_size]

        # --- Save
        tr_X = np.concatenate([tr_anchor, tr_positive], axis=1)
        pickle.dump([tr_X, tr_y],
                    open(fi_out + "For combination/" + "L" + str(loop_i) + "_Data_train_from_tripletnet1_mi" + str(ix) + ".pkl", "wb"))
        return

    # -----main train_triplet1
    triplets = gen_triplet_idx(loop_i, tile, ix)
    xuly(triplets, loop_i, ix)
    return


# %%
def train_triplet2(loop_i, ix):
    def gen_triplet_idx(loop_i, tile):
        pair_trP, pair_trN = get_pair(ix)
        # np.random.seed(2022)
        np.random.seed()
        triplets = []
        for dis_j in range(di_num):  # A.shape[1]
            mir_link_dis_j = pair_trP['mir'][pair_trP['dis'] == dis_j]
            mir_nolink_dis_j = pair_trN['mir'][pair_trN['dis'] == dis_j]
            for mir_link in mir_link_dis_j:
                np.random.shuffle(mir_nolink_dis_j)

                if (tile == -1) or (len(mir_nolink_dis_j) < tile):
                    tile = len(mir_nolink_dis_j)
                for mir_nolink in mir_nolink_dis_j[:tile]:
                    triplets.append((dis_j, mir_link, mir_nolink))

        pickle.dump({'triplets': triplets},  # save triplets
                    open(fi_out + "Triplet_sample_train/" + "L" + str(loop_i) + "_From_tripletnet2_mi" + str(ix) + ".pkl", "wb"))
        print('triplets.shape', len(triplets))
        return triplets

    def tripletNET_2():
        in1 = keras_layers.Input(didim)
        in2 = keras_layers.Input(midim)
        in3 = keras_layers.Input(midim)

        f_di = diNET()
        f_mi1 = miNET()
        f_mi2 = miNET()

        di = f_di(in1)
        mi1 = f_mi1(in2)
        mi2 = f_mi1(in3)

        in1A = keras_layers.Input(64)
        in2A = keras_layers.Input(64)
        in3A = keras_layers.Input(64)

        diA, mi1A, mi2A = in1A, in2A, in3A

        di = keras_layers.Concatenate()([di, diA])
        mi1 = keras_layers.Concatenate()([mi1, mi1A])
        mi2 = keras_layers.Concatenate()([mi2, mi2A])

        out = keras_layers.Concatenate()([di, mi1, mi2])

        final = keras_models.Model(inputs=[in1, in2, in3, in1A, in2A, in3A], outputs=out)
        return final

    def get_sample_triplet2(full_pair, train_or_test):  # train_or_test = [0, 1] (train) / [3, 4] (test)
        pair_N = np.argwhere(full_pair.values == train_or_test[0])
        pair_P = np.argwhere(full_pair.values == train_or_test[1])

        pair_N = {'mir': pair_N[:, 0], 'dis': pair_N[:, 1]}
        pair_P = {'mir': pair_P[:, 0], 'dis': pair_P[:, 1]}

        triplets, y = [], []
        for mir_i in range(mi_num):
            dis_link_mir_i = pair_P['dis'][pair_P['mir'] == mir_i]
            dis_nolink_mir_i = pair_N['dis'][pair_N['mir'] == mir_i]

            for dis_link in dis_link_mir_i:
                triplets.append((dis_link, mir_i, mir_i))
                y.append(1)

            for dis_nolink in dis_nolink_mir_i:
                triplets.append((dis_nolink, mir_i, mir_i))
                y.append(0)
        return triplets, y

    def xuly(triplets, loop_i, ix):
        miRNA_embeddings = miRNA_auto_encoder(misim_data)  # Dữ liệu miRNA
        miRNA_embeddings = miRNA_auto_encoder2(miRNA_embeddings)
        miRNA_embeddings = miRNA_auto_encoder3(miRNA_embeddings)

        disease_embeddings = dis_auto_encoder(disim_data)  # Dữ liệu disease
        disease_embeddings = dis_auto_encoder2(disease_embeddings)
        disease_embeddings = dis_auto_encoder3(disease_embeddings)
        # (1)
        idx = np.arange(len(triplets))
        # np.random.seed(2022)
        np.random.seed()
        np.random.shuffle(idx)

        triplets = np.array(triplets)
        tr_di = disim_data.iloc[triplets[idx, 0]]
        tr_mi1 = misim_data.iloc[triplets[idx, 1]]
        tr_mi2 = misim_data.iloc[triplets[idx, 2]]
        print('tr_di.shape, tr_mi1.shape, tr_mi2.shape', tr_di.shape, tr_mi1.shape, tr_mi2.shape)

        tr_diA = disease_embeddings[triplets[idx, 0], :]
        tr_mi1A = miRNA_embeddings[triplets[idx, 1], :]
        tr_mi2A = miRNA_embeddings[triplets[idx, 2], :]

        # (2) #Train tripletNET_2
        print("\n--- Train tripletNET_2 ...")
        y = np.array([0] * len(triplets))  # mảng số 0, để khớp code
        tripletnet2 = tripletNET_2()
        tripletnet2.compile(loss=triplet_loss2, optimizer='adam')
        _ = tripletnet2.fit([tr_di, tr_mi1, tr_mi2, tr_diA, tr_mi1A, tr_mi2A], y, epochs=ne, verbose=2)
        print("Done")

        # (3) #GET TEST TRIPLETNET + PREDICT NO USE XGBOOST
        te_triplets, te_y = get_sample_triplet2(full_pair4, [3, 4])
        te_triplets = np.array(te_triplets)
        te_di = disim_data.iloc[te_triplets[:, 0]]
        te_mi1 = misim_data.iloc[te_triplets[:, 1]]
        te_mi2 = misim_data.iloc[te_triplets[:, 2]]

        te_diA = disease_embeddings[te_triplets[:, 0], :]
        te_mi1A = miRNA_embeddings[te_triplets[:, 1], :]
        te_mi2A = miRNA_embeddings[te_triplets[:, 2], :]

        te_distance = tripletnet2.predict([te_di, te_mi1, te_mi2, te_diA, te_mi1A, te_mi2A])

        anchor = te_distance[:, :diemb_size]
        positive = te_distance[:, diemb_size:diemb_size + miemb_size]
        negative = te_distance[:, diemb_size + miemb_size:]

        # -- Save
        te_X = np.concatenate([anchor, positive], axis=1)
        pickle.dump([te_X, te_y], open(fi_out + "Data_test/" + "L" + str(loop_i) + "_From_tripletnet2_mi" + str(ix) + ".pkl", "wb"))

        # (4) #GET TRAIN SET FOR TRADITIONAL
        # print("\n\n--- Lay Train cho traditional, tripletnet2")
        tr_triplets, tr_y = get_sample_triplet2(full_pair4, [0, 1])
        # print(len(tr_triplets))
        # print(len(tr_y))

        idx = np.arange(len(tr_triplets))
        tr_triplets = np.array(tr_triplets)
        tr_di = disim_data.iloc[tr_triplets[idx, 0]]
        tr_mi1 = misim_data.iloc[tr_triplets[idx, 1]]
        tr_mi2 = misim_data.iloc[tr_triplets[idx, 2]]

        tr_diA = disease_embeddings[tr_triplets[idx, 0], :]
        tr_mi1A = miRNA_embeddings[tr_triplets[idx, 1], :]
        tr_mi2A = miRNA_embeddings[tr_triplets[idx, 2], :]

        tr_distance = tripletnet2.predict([tr_di, tr_mi1, tr_mi2, tr_diA, tr_mi1A, tr_mi2A])

        tr_anchor = tr_distance[:, :diemb_size]
        tr_positive = tr_distance[:, diemb_size:diemb_size + miemb_size]

        # --- Save
        tr_X = np.concatenate([tr_anchor, tr_positive], axis=1)
        pickle.dump([tr_X, tr_y],
                    open(fi_out + "For combination/" + "L" + str(loop_i) + "_Data_train_from_tripletnet2_mi" + str(ix) + ".pkl", "wb"))
        return

    # -----main train_triplet2
    triplets = gen_triplet_idx(loop_i, tile)
    xuly(triplets, loop_i, ix)
    return


# %%
def read_data(triplet_i, loop_i, ix):
    print("--- READ TRAIN SET ---")
    tr_X, tr_y = pickle.load(open(
        fi_out + "For combination/L" + str(loop_i) + "_Data_train_from_tripletnet" + str(
            triplet_i) + "_mi" + str(ix) + ".pkl", "rb"))
    tr_y = np.array(tr_y)
    print('tr_X.shape', np.array(tr_X).shape)
    print(tr_y.shape)

    print("--- READ TEST SET ---")
    te_X, te_y = pickle.load(open(fi_out + "Data_test/L" + str(loop_i) + "_From_tripletnet" + str(
        triplet_i) + "_mi" + str(ix) + ".pkl", "rb"))
    te_y = np.array(te_y)
    print('te_X.shape', np.array(te_X).shape)
    print(te_y.shape)
    return tr_X, tr_y, te_X, te_y


def get_test_score(yprob, ypred, ytrue):
    acc = accuracy_score(ytrue, ypred)
    pre = precision_score(ytrue, ypred)
    auc_ = roc_auc_score(ytrue, yprob)  # ko đặt trùng tên auc, mà phải auc_
    precision, recall, _ = precision_recall_curve(ytrue, yprob)
    aupr_ = auc(recall, precision)
    return acc, pre, auc_, aupr_


# %%
def main_combine(loop_i, ix):
    def get_yprob_ypred(tr_X, tr_y, te_X, te_y, triplet_i, loop_i):
        model = XGBClassifier(booster='gbtree', n_jobs=2, learning_rate=lrr, n_estimators=xgne, random_state=48)
        model.fit(tr_X, tr_y)
        prob_y = model.predict_proba(te_X)[:, 1]
        pred_y = model.predict(te_X)

        np.savetxt(fi_out + 'Results/Combination/L' + str(loop_i) + '_prob_' + 'X_triplet' \
                   + str(triplet_i) + "_mi" + str(ix) + ".csv", prob_y)
        np.savetxt(fi_out + 'Results/Combination/L' + str(loop_i) + '_true_' + 'X_triplet' \
                   + str(triplet_i) + "_mi" + str(ix) + ".csv", te_y, fmt="%d")
        return prob_y, pred_y

    # -------------MAIN OF main_combine--------------
    print('Triplet 1')
    tr_X_1, tr_y_1, te_X_1, te_y_1 = read_data(1, loop_i, ix)

    prob_y_trp1_m, pred_y_trp1_m = get_yprob_ypred(tr_X_1, tr_y_1, te_X_1, te_y_1, \
                                                   1, loop_i)  # test

    # acc_tr1_m, pre_tr1_m, auc_tr1_m, aupr_tr1_m = get_test_score(prob_y_trp1_m, pred_y_trp1_m, te_y_1)

    print('Triplet 2')
    tr_X_2, tr_y_2, te_X_2, te_y_2 = read_data(2, loop_i, ix)

    prob_y_trp2_m, pred_y_trp2_m = get_yprob_ypred(tr_X_2, tr_y_2, te_X_2, te_y_2, 2, \
                                                   loop_i)
    # acc_tr2_m, pre_tr2_m, auc_tr2_m, aupr_tr2_m = get_test_score(prob_y_trp2_m, pred_y_trp2_m, te_y_1)

    return

def main4(method, loop_i, ix):
    ACC, PRE, AUC, AUPR = [], [], [], []
    name_trb_true = 'L' + str(loop_i) + '_true_trbX_mi' + str(ix)
    name_trb_prob = 'L' + str(loop_i) + '_prob_trbX_mi' + str(ix)

    print('Triplet 1')
    tr_X_1, tr_y_1, te_X_1, te_y_1 = read_data(1, loop_i, ix)
    prob_y_trp1_m = np.genfromtxt(fi_out + 'Results/Combination/L' + str(loop_i) + '_prob_' + \
                                  'X_triplet1' + '_mi' + str(ix) + '.csv')

    print('Triplet 2')
    tr_X_2, tr_y_2, te_X_2, te_y_2 = read_data(2, loop_i, ix)
    prob_y_trp2_m = np.genfromtxt(fi_out + 'Results/Combination/L' + str(loop_i) + '_prob_' + \
                                   'X_triplet2' + '_mi' + str(ix) + '.csv')

    # error_1 và error_2 là sai số (hoặc độ không chắc chắn) của mỗi mô hình
    error_1 = np.abs(te_y_1 - prob_y_trp1_m).mean()  # Sai số của model1
    error_2 = np.abs(te_y_2 - prob_y_trp2_m).mean()  # Sai số của model2

    # Tính trọng số nghịch đảo của sai số
    weight1 = 1 / error_1
    weight2 = 1 / error_2

    # Kết hợp dự đoán dựa trên trọng số từ sai số
    prob_y_m = (prob_y_trp1_m * weight1 + prob_y_trp2_m * weight2) / (weight1 + weight2)

    if method == "M":  # MLR
        pred_y_m = np.array(prob_y_m > 0)
    else:
        pred_y_m = np.array(prob_y_m >= 0.5)

    np.savetxt(fi_out + 'Results/Combination/' + name_trb_true + '.csv', te_y_1, fmt='%d')
    np.savetxt(fi_out + 'Results/Combination/' + name_trb_prob + '.csv', prob_y_m)

    acc_m, pre_m, auc_m, aupr_m = get_test_score(prob_y_m, pred_y_m, te_y_1)
    print(auc_m, aupr_m)

    ACC.append(acc_m)
    PRE.append(pre_m)
    AUC.append(auc_m)
    AUPR.append(aupr_m)

    return np.mean(np.array(AUC)), np.mean(np.array(AUPR))


# %%
def main():

    global full_pair4
    global misim_data
    global disim_data

    AUC_all = []
    AUPR_all = []
    mi_set = np.genfromtxt(fi_A + 'mi_set.csv', delimiter = ',').astype(int)
    for mi_k in mi_set:
        print('MIRNA', mi_k,'*'*50)
        full_pair4 = pd.read_csv(fi_A + 'y_4loai_mi' + str(mi_k) + '.csv', header=None)  # A fold begin from 1
        misim_data = pd.read_csv(fi_feature + 'SR_FS_mi' + str(mi_k) + '.csv', header=None)
        disim_data = pd.read_csv(fi_feature + 'SD_SS_mi' + str(mi_k) + '.csv', header=None)

        for loop_i in range(bgl, nloop + 1):
            #######(1)######
            print('Loop ', loop_i)
            print('Gen prob of each triplet')
            train_triplet1(loop_i, mi_k)
            train_triplet2(loop_i, mi_k)
            main_combine(loop_i, mi_k)
            AUC_l, AUPR_l = main4('X', loop_i, mi_k)
            AUC_all.append(AUC_l)
            AUPR_all.append(AUPR_l)
            #######(2)######
    
    print('AUC final mean:', np.mean(np.array(AUC_all)))
    print('AUPR final mean:', np.mean(np.array(AUPR_all)))

if __name__ == "__main__":
    main()


