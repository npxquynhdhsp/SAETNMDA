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
nm, nd = 788, 374  # @Q mo rong 811, 557
di_num, mi_num = nd, nm
didim, midim = nd, nm
l_alpha = 1
fi_A = '../IN/INDE_TEST/'
fi_feature = '../IN/INDE_TEST/'
fi_out = './OUT Q_INDE/'

temp = '_H2all'
# temp = ''

full_pair4 = pd.read_csv(fi_A + 'y_4loai_Inde' + temp + '.csv', header=None)  # A fold begin from 1
misim_data = pd.read_csv(fi_feature + 'Q18.3_IM_Inde' + temp + '.csv', header=None)
disim_data = pd.read_csv(fi_feature + 'Q18.3_ID_Inde' + temp + '.csv', header=None)


# %%
def get_pair():  # get pair indexes from y_4loai
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
    autoencoder.fit(x_train, x_train, epochs=500, batch_size=64, shuffle=True)
    miRNA_encoded = encoder.predict(x_train)
    return miRNA_encoded

def miRNA_auto_encoder2(x_train):
    input = layers.Input(512)
    encoded = layers.Dense(256, activation='elu')(input)
    decoded = layers.Dense(512, activation='elu')(encoded)
    autoencoder = models.Model(inputs=input, outputs=decoded)
    encoder = models.Model(inputs=input, outputs=encoded)

    adam = Adam(learning_rate=0.001)
    autoencoder.compile(optimizer=adam, loss='mse')
    autoencoder.fit(x_train, x_train, epochs=500, batch_size=64, shuffle=True)
    miRNA_encoded = encoder.predict(x_train)
    return miRNA_encoded

def miRNA_auto_encoder3(x_train):
    input = layers.Input(256)
    encoded = layers.Dense(64, activation='elu')(input)
    decoded = layers.Dense(256, activation='elu')(encoded)
    # 构建自编码模型
    autoencoder = models.Model(inputs=input, outputs=decoded)
    encoder = models.Model(inputs=input, outputs=encoded)
    # 激活模型
    adam = Adam(learning_rate=0.001)
    autoencoder.compile(optimizer=adam, loss='mse')
    autoencoder.fit(x_train, x_train, epochs=500, batch_size=64, shuffle=True)
    miRNA_encoded = encoder.predict(x_train)
    return miRNA_encoded

def dis_auto_encoder(x_train):
    input = layers.Input(didim)
    encoded = layers.Dense(512, activation='elu')(input)
    decoded = layers.Dense(didim, activation='elu')(encoded)
    autoencoder = models.Model(inputs=input, outputs=decoded)
    encoder = models.Model(inputs=input, outputs=encoded)

    adam = Adam(learning_rate=0.001)
    autoencoder.compile(optimizer=adam, loss='mse')
    autoencoder.fit(x_train, x_train, epochs=500, batch_size=64, shuffle=True)
    miRNA_encoded = encoder.predict(x_train)
    return miRNA_encoded

def dis_auto_encoder2(x_train):
    input = layers.Input(512)
    encoded = layers.Dense(256, activation='elu')(input)
    decoded = layers.Dense(512, activation='elu')(encoded)
    autoencoder = models.Model(inputs=input, outputs=decoded)
    encoder = models.Model(inputs=input, outputs=encoded)

    adam = Adam(learning_rate=0.001)
    autoencoder.compile(optimizer=adam, loss='mse')
    autoencoder.fit(x_train, x_train, epochs=500, batch_size=64, shuffle=True)
    miRNA_encoded = encoder.predict(x_train)
    return miRNA_encoded

def dis_auto_encoder3(x_train):
    input = layers.Input(256)
    encoded = layers.Dense(64, activation='elu')(input)
    decoded = layers.Dense(256, activation='elu')(encoded)
    autoencoder = models.Model(inputs=input, outputs=decoded)
    encoder = models.Model(inputs=input, outputs=encoded)

    adam = Adam(learning_rate=0.001)
    autoencoder.compile(optimizer=adam, loss='mse')
    autoencoder.fit(x_train, x_train, epochs=500, batch_size=64, shuffle=True)
    miRNA_encoded = encoder.predict(x_train)
    return miRNA_encoded

def triplet_loss1(y_true, y_pred, miemb_size = miemb_size, diemb_size = diemb_size):
    def loss_(y_true, y_pred):
        anchor = y_pred[:, :miemb_size]
        positive = y_pred[:, miemb_size:miemb_size + diemb_size]
        negative = y_pred[:, miemb_size + diemb_size:]

        positive_dist = tf.square(anchor - positive)
        negative_dist = tf.square(anchor - negative)

        l1 = tf.maximum(positive_dist - negative_dist + alpha, 0.)
        l2 = tf.maximum(0., alpha - positive_dist + negative_dist)  # Margin Ranking Loss
        return tf.reduce_mean(l1) * l_alpha + tf.reduce_mean(l2) * (1 - l_alpha)
    return loss_(y_true, y_pred)

def triplet_loss2(y_true, y_pred, miemb_size = miemb_size, diemb_size = diemb_size):
    def loss_(y_true, y_pred):
        anchor = y_pred[:, :diemb_size]
        positive = y_pred[:, diemb_size:diemb_size + miemb_size]
        negative = y_pred[:, diemb_size + miemb_size:]

        positive_dist = tf.square(anchor - positive)
        negative_dist = tf.square(anchor - negative)

        l1 = tf.maximum(positive_dist - negative_dist + alpha, 0.)
        l2 = tf.maximum(0., alpha - positive_dist + negative_dist)
        return tf.reduce_mean(l1) * l_alpha + tf.reduce_mean(l2) * (1 - l_alpha)
    return loss_(y_true, y_pred)


# %%
def train_triplet1(loop_i):
    def gen_triplet_idx(loop_i, tile):
        pair_trP, pair_trN = get_pair()
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
                    open(fi_out + "Triplet_sample_train/" + "L" + str(loop_i) + "_From_tripletnet1.pkl", "wb"))
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

    def xuly(triplets, loop_i):
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
        pickle.dump([te_X, te_y], open(fi_out + "Data_test/" + "L" + str(loop_i) + "_From_tripletnet1.pkl", "wb"))

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
                    open(fi_out + "For combination/" + "L" + str(loop_i) + "_Data_train_from_tripletnet1.pkl", "wb"))
        return

    # -----main train_triplet1
    triplets = gen_triplet_idx(loop_i, tile=tile)
    xuly(triplets, loop_i)
    return


# %%
def train_triplet2(loop_i):
    def gen_triplet_idx(loop_i, tile):
        pair_trP, pair_trN = get_pair()
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
                    open(fi_out + "Triplet_sample_train/" + "L" + str(loop_i) + "_From_tripletnet2.pkl", "wb"))
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

    def xuly(triplets, loop_i):
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
        pickle.dump([te_X, te_y], open(fi_out + "Data_test/" + "L" + str(loop_i) + "_From_tripletnet2.pkl", "wb"))

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
                    open(fi_out + "For combination/" + "L" + str(loop_i) + "_Data_train_from_tripletnet2.pkl", "wb"))
        return

    # -----main train_triplet2
    triplets = gen_triplet_idx(loop_i, tile=tile)
    xuly(triplets, loop_i)
    return


# %%
def read_data(triplet_i, loop_i):
    print("--- READ TRAIN SET ---")
    tr_X, tr_y = pickle.load(open(
        fi_out + "For combination/L" + str(loop_i) + "_Data_train_from_tripletnet" + str(
            triplet_i) + ".pkl", "rb"))
    tr_y = np.array(tr_y)
    print('tr_X.shape', np.array(tr_X).shape)
    print(tr_y.shape)

    print("--- READ TEST SET ---")
    te_X, te_y = pickle.load(open(fi_out + "Data_test/L" + str(loop_i) + "_From_tripletnet" + str(
        triplet_i) + ".pkl", "rb"))
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
def main_combine(loop_i):
    def get_yprob_ypred(tr_X, tr_y, te_X, te_y, triplet_i, loop_i):
        ##CHU Y METHOD DE LAY PRED (2 CHO)
        # --------------------------------------------------
        model = XGBClassifier(booster='gbtree', n_jobs=2, learning_rate=lrr, n_estimators=xgne, random_state=48)
        model.fit(tr_X, tr_y)
        prob_y = model.predict_proba(te_X)[:, 1]
        pred_y = model.predict(te_X)

        np.savetxt(fi_out + 'Results/Combination/L' + str(loop_i) + '_prob_' + 'X_triplet' \
                   + str(triplet_i) + '.csv', prob_y)
        np.savetxt(fi_out + 'Results/Combination/L' + str(loop_i) + '_true_' + 'X_triplet' \
                   + str(triplet_i) + '.csv', te_y, fmt="%d")
        return prob_y, pred_y

    # -------------MAIN OF main_combine--------------
    print('Triplet 1')
    tr_X_1, tr_y_1, te_X_1, te_y_1 = read_data(1, loop_i)

    prob_y_trp1_m, pred_y_trp1_m = get_yprob_ypred(tr_X_1, tr_y_1, te_X_1, te_y_1, \
                                                   1, loop_i)  # test

    # acc_tr1_m, pre_tr1_m, auc_tr1_m, aupr_tr1_m = get_test_score(prob_y_trp1_m, pred_y_trp1_m, te_y_1)

    print('Triplet 2')
    tr_X_2, tr_y_2, te_X_2, te_y_2 = read_data(2, loop_i)

    prob_y_trp2_m, pred_y_trp2_m = get_yprob_ypred(tr_X_2, tr_y_2, te_X_2, te_y_2, 2, \
                                                   loop_i)
    # acc_tr2_m, pre_tr2_m, auc_tr2_m, aupr_tr2_m = get_test_score(prob_y_trp2_m, pred_y_trp2_m, te_y_1)

    return

def main_combine4(loop_i, method):# thêm trọng số nghịch đảo
    ACC, PRE, AUC, AUPR = [], [], [], []
    name_trb_true = 'L' + str(loop_i) + '_true_trbX'
    name_trb_prob = 'L' + str(loop_i) + '_prob_trbX'

    print('Triplet 1')
    tr_X_1, tr_y_1, te_X_1, te_y_1 = read_data(1, loop_i)
    prob_y_trp1_m = np.genfromtxt(fi_out + 'Results/Combination/L' + str(loop_i) + '_prob_' + \
                                  'X_triplet1.csv')

    print('Triplet 2')
    tr_X_2, tr_y_2, te_X_2, te_y_2 = read_data(2, loop_i)
    prob_y_trp2_m = np.genfromtxt(fi_out + 'Results/Combination/L' + str(loop_i) + '_prob_' + \
                                   'X_triplet2.csv')

    # error_1 và error_2 là sai số (hoặc độ không chắc chắn) của mỗi mô hình
    error_1 = np.abs(te_y_1 - prob_y_trp1_m).mean()  # Sai số của model1
    error_2 = np.abs(te_y_2 - prob_y_trp2_m).mean()  # Sai số của model2

    # Tính trọng số nghịch đảo của sai số
    weight1 = 1 / error_1
    weight2 = 1 / error_2

    # Kết hợp dự đoán dựa trên trọng số từ sai số
    prob_y_m = (prob_y_trp1_m * weight1 + prob_y_trp2_m * weight2) / (weight1 + weight2)

    # for j in range(10):
    #     print(round(prob_y_trp1_m[j], 4), round(prob_y_trp2_m[j], 4), round(prob_y_m[j], 4))

    if (method == "M") or (method == "L"):  # Regression
        pred_y_m = np.array(prob_y_m > 0)
    else:
        pred_y_m = np.array(prob_y_m >= 0.5)

    np.savetxt(fi_out + 'Results/Combination/' + name_trb_true + '.csv', te_y_1, fmt='%d')
    np.savetxt(fi_out + 'Results/Combination/' + name_trb_prob + '.csv', prob_y_m)

    acc_m, pre_m, auc_m, aupr_m = get_test_score(prob_y_m, pred_y_m, te_y_1)
    print('AUC, AUPR', auc_m, aupr_m)

    ACC.append(acc_m)
    PRE.append(pre_m)
    AUC.append(auc_m)
    AUPR.append(aupr_m)

    return np.mean(np.array(AUC)), np.mean(np.array(AUPR))

# %%
def join2_prob(_folder, _name1, _name2):  # @Q join loop
    PROB_all = []
    TRUE_all = []
    for loop_i in range(1, nloop + 1):
        prob_i = np.genfromtxt(_folder + 'L' + str(loop_i) + _name1 + '.csv')
        true_i = np.genfromtxt(_folder + 'L' + str(loop_i) + _name2 + '.csv').astype(int)
        PROB_all.extend(prob_i)
        TRUE_all.extend(true_i)
    np.savetxt(_folder + 'PROB_all.csv', np.array(PROB_all))
    np.savetxt(_folder + 'TRUE_all.csv', np.array(TRUE_all), fmt='%d')
    return


# %%
def danhgia_rieng():
    def gen_setvalue(path):
        arr_y_loai = []
        arr_y_pred = []
        path_loai_i = path + '.csv'
        arr_y_loai_i = np.genfromtxt(path_loai_i)
        arr_y_loai.append(arr_y_loai_i)
        arr_y_pred_i = np.where(arr_y_loai_i >= 0.5, 1, 0)  # @ -1 hay 0 thì phải chỉnh ở đây
        arr_y_pred.append(arr_y_pred_i)
        return arr_y_loai, arr_y_pred

    def draw_plot_KFOLD(plt, arr_y_test, arr_y_prob, name_file):
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(11, 4))
        ax_roc = ax[0]
        ax_rpc = ax[1]

        n_samples = 0
        for i, y_test in enumerate(arr_y_test):
            n_samples += y_test.shape[0]
        mean_fpr = np.linspace(0, 1, n_samples)
        roc_aucs = []
        tprs = []

        mean_rec = np.linspace(0, 1, n_samples)
        pres = []
        rpc_aucs = []

        # get fpr, tpr scores
        for i, (y_test, y_prob) in enumerate(zip(arr_y_test, arr_y_prob)):
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            # plot ROC curve
            if len(arr_y_test) > 1:  # 5fold
                ax_roc.plot(fpr, tpr, lw=1, alpha=0.5, label='ROC fold %d (AUC = %0.4f)' % (i + 1, roc_auc))  # @@@Q

            interp_tpr = interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            roc_aucs.append(roc_auc)

        ax_roc.plot([0, 1], [0, 1], lw=2, alpha=.8, linestyle='--', color='r', label='Chance')  # @Q

        # Ve ROC mean
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_roc_auc = auc(mean_fpr, mean_tpr)
        std__roc_auc = np.std(roc_aucs)

        if len(arr_y_test) > 1:
            ax_roc.plot(mean_fpr, mean_tpr, color='b',
                        label=r'Mean ROC (AUC = %0.4f $\pm$ %0.3f)' % (mean_roc_auc, std__roc_auc),
                        lw=2, alpha=.8)
        else:
            ax_roc.plot(mean_fpr, mean_tpr, color='b',
                        label=r'Mean ROC (AUC = %0.4f)' % (mean_roc_auc),
                        lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax_roc.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2)  # ,label=r'$\pm$ 1 std. dev.')

        # Dat ten
        ax_roc.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
        ax_roc.set_title(label="ROC curve", fontweight='bold')
        ax_roc.set_xlabel('False Positive Rate', fontweight='bold')
        ax_roc.set_ylabel('True Positive Rate', fontweight='bold')
        ax_roc.legend(fontsize='x-small', loc='lower right',
                      bbox_to_anchor=(1, 0.05))  # tính từ toạ độ (1,0.05), legend nằm ở lower right

        # get precision, recall scores

        for i, (y_test, y_prob) in enumerate(zip(arr_y_test, arr_y_prob)):
            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            rpc_aupr = average_precision_score(y_test, y_prob)
            # plot precision recall curve
            # @Q vẽ 5 fold
            if len(arr_y_test) > 1:
                ax_rpc.plot(recall, precision, lw=1, alpha=0.5,
                            label='PR fold %d (AP = %0.4f)' % (i + 1, rpc_aupr))  # @@

            interp_pre = interp(mean_rec, recall, precision)
            # interp_tpr[0] = 0.0
            pres.append(interp_pre)
            rpc_aucs.append(rpc_aupr)

        y_tests = np.array([])
        for y_test in arr_y_test:
            y_tests = np.hstack((y_tests, y_test.ravel()))

        no_skill = len(y_tests[y_tests == 1]) / y_tests.shape[0]
        ax_rpc.plot([0, 1], [no_skill, no_skill], linestyle='--', lw=2, color='r', label='Chance')

        # Ve duong mean
        all_y_test = np.concatenate(arr_y_test)
        all_y_prob = np.concatenate(arr_y_prob)
        precision, recall, _ = precision_recall_curve(all_y_test, all_y_prob)

        # --- Lay TB cac fold
        ave = []
        for i in range(len(arr_y_test)):
            ave.append(average_precision_score(arr_y_test[i], arr_y_prob[i]))

        if len(arr_y_test) > 1:
            std__pr_aupr = np.std(rpc_aucs)
            ax_rpc.plot(recall, precision, color='b',
                        label=r'Mean PR (AUPR = %0.4f $\pm$ %0.3f)' %
                              (average_precision_score(all_y_test, all_y_prob), std__roc_auc),  # Cu
                        #  (np.mean(ave),np.std(ave)), # moi
                        lw=2, alpha=1)
        else:
            ax_rpc.plot(recall, precision, color='b',
                        label=r'Mean PR (AUPR = %0.4f)' %
                              (average_precision_score(all_y_test, all_y_prob)),  # CU
                        #  (np.mean(ave)), # moi
                        lw=2, alpha=1)

        # Dat ten
        ax_rpc.set_title('Precision-Recall Curve', fontweight='bold')
        ax_rpc.set_xlabel('Recall', fontweight='bold')
        ax_rpc.set_ylabel('Precision', fontweight='bold')
        ax_rpc.legend(fontsize='x-small', loc='lower left', bbox_to_anchor=(0.05, 0.05))

        # return plt

        plt.savefig(name_file + '.pdf', format='pdf', bbox_inches='tight')
        plt.savefig(name_file + '.png', format='png', dpi=300, bbox_inches='tight')
        plt.show()
        return

    # ------MAIN of danhgiarieng-------
    path_eval = fi_out + 'Results/Combination/'
    str_fi1 = 'PROB_all'
    str_fi2 = 'TRUE_all'
    arr_y_prob, arr_y_pred = gen_setvalue(path_eval + str_fi1)
    arr_y_test, arr_y_pred_temp = gen_setvalue(path_eval + str_fi2)

    path_out_fig = path_eval + 'FIG/'
    name_plot = 'AUC_AUPR_Inde'
    fig_file = path_out_fig + name_plot
    draw_plot_KFOLD(plt, arr_y_test, arr_y_prob, fig_file)
    return


# %%
def main():
    AUC_all = []
    AUPR_all = []
    method = 'X'
    print('................METHOD.................', method)
    for loop_i in range(bgl, nloop + 1):
        #######(1)######
        print('Loop ', loop_i)
        print('Gen prob of each triplet')
        train_triplet1(loop_i)
        train_triplet2(loop_i)
        main_combine(loop_i)
        AUC_l, AUPR_l = main_combine4(loop_i, method)
        AUC_all.append(AUC_l)
        AUPR_all.append(AUPR_l)
        #######(2)######
    
    print('AUC final mean:', np.mean(np.array(AUC_all)))
    print('AUPR final mean:', np.mean(np.array(AUPR_all)))
    

if __name__ == "__main__":
    main()


