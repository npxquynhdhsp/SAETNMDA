# %%
from params import args
import time
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as keras_layers
import tensorflow.keras.models as keras_models
import pickle
from sklearn.metrics import auc, roc_auc_score, accuracy_score, precision_recall_curve, precision_score, \
confusion_matrix
from models import triplet_loss1, get_sample_triplet1
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

# %%
def get_pair(args, fold, loop_i): #get pair indexes from y_4loai_fold
    full_pair = pd.read_csv(args.fi_A + 'L' + str(loop_i) + '/y_4loai_fold' + str(fold+1) +'.csv', header=None) #A fold begin from 1
    pair_trN = np.argwhere(full_pair.values == 0)
    pair_trP = np.argwhere(full_pair.values == 1)
    #-
    if args.cb == 1:
        #--- 1 trong 2
        np.random.seed(2022)
        random_indices = np.random.choice(pair_trN.shape[0], size=pair_trP.shape[0], replace=False)
        pair_trN = pair_trN[random_indices]
        #-
        # train_pair_list = np.genfromtxt(args.fi_train_test + 'L' + str(loop_i) + '/train_pairs_1a1d_fold' + str(fold) + '.csv',
        #                                 delimiter=',').astype(int)
        # train_pair_lbl = np.genfromtxt(args.fi_train_test + 'L' + str(loop_i) + '/ytrain_1a1d_fold' + str(fold) + '.csv').astype(int).reshape(-1)
        # pair_trP = train_pair_list[train_pair_lbl == 1]
        # pair_trN = train_pair_list[train_pair_lbl == 0]
        #---
    #-
    # pair_teN = np.argwhere(full_pair.values == 3)
    # pair_teP = np.argwhere(full_pair.values == 4)
    print('pair_trainP.shape', pair_trP.shape)
    print('pair_trainN.shape', pair_trN.shape)
    # print('pair_testP.shape', pair_teP.shape)
    # print('pair_testN.shape', pair_teN.shape)
    pair_trN = {'mir': pair_trN[:, 0], 'dis': pair_trN[:, 1]}
    pair_trP = {'mir': pair_trP[:, 0], 'dis': pair_trP[:, 1]}
    # pair_teN = {'mir': pair_teN[:, 0], 'dis': pair_teN[:, 1]}
    # pair_teP = {'mir': pair_teP[:, 0], 'dis': pair_teP[:, 1]}
    # return pair_trP, pair_trN, pair_teP, pair_teN
    return pair_trP, pair_trN

# %%
def gen_triplet_idx(args, fold, loop_i): 
    # pair_trP, pair_trN, _, _ = get_pair(args, fold)
    pair_trP, pair_trN = get_pair(args, fold, loop_i)
    np.random.seed(2022)
    triplets = []
    for mir_i in range(args.mi_num): #A.shape[0]
        dis_link_mir_i = pair_trP['dis'][pair_trP['mir'] == mir_i] 
        dis_nolink_mir_i = pair_trN['dis'][pair_trN['mir'] == mir_i] 
        for dis_link in dis_link_mir_i:
            np.random.shuffle(dis_nolink_mir_i) 

            if (args.tile == -1) or (len(dis_nolink_mir_i) < args.tile): 
                args.tile = len(dis_nolink_mir_i)
            for dis_nolink in dis_nolink_mir_i[:args.tile]: 
                triplets.append((mir_i, dis_link, dis_nolink))

    pickle.dump({'triplets': triplets}, 
                open(args.fi_out + "Triplet_sample_train/" + "L" + str(loop_i) + "_From_tripletnet1_fold" + str(fold) + ".pkl","wb"))
    print('triplets.shape',len(triplets)) 
    return triplets

# %%
def miNET(args):
    net = keras_models.Sequential()
    net.add(keras_layers.Dense(256))
    net.add(keras_layers.Dense(64))
    # net.add(keras_layers.Dense(args.miemb_size))
    # net.add(keras_layers.Lambda(lambda x: l2_normalize(x, axis=1)))
    return net

def diNET(args):
    net = keras_models.Sequential()
    net.add(keras_layers.Dense(256))
    net.add(keras_layers.Dense(64))
    # net.add(keras_layers.Dense(args.diemb_size))
    # net.add(keras_layers.Lambda(lambda x: l2_normalize(x, axis=1)))
    return net

def miRNA_auto_encoder(x_train):
    input = layers.Input(args.midim)
    encoded = layers.Dense(512, activation='elu')(input)
    decoded = layers.Dense(args.midim, activation='elu')(encoded)
    autoencoder = models.Model(inputs=input, outputs=decoded)
    encoder = models.Model(inputs=input, outputs=encoded)

    adam = Adam(learning_rate=0.001)
    autoencoder.compile(optimizer=adam, loss='mse')
    autoencoder.fit(x_train, x_train, epochs=500, batch_size=64, shuffle=True)
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
    input = layers.Input(args.didim)
    encoded = layers.Dense(512, activation='elu')(input)
    decoded = layers.Dense(args.didim, activation='elu')(encoded)
    autoencoder = models.Model(inputs=input, outputs=decoded)
    encoder = models.Model(inputs=input, outputs=encoded)

    adam = Adam(learning_rate=0.001)
    autoencoder.compile(optimizer=adam, loss='mse')
    autoencoder.fit(x_train, x_train, epochs=500, batch_size=64, shuffle=True)
    disease_encodeds = encoder.predict(x_train)
    return disease_encodeds

def dis_auto_encoder2(x_train):
    input = layers.Input(512)
    encoded = layers.Dense(256, activation='elu')(input)
    decoded = layers.Dense(512, activation='elu')(encoded)
    autoencoder = models.Model(inputs=input, outputs=decoded)
    encoder = models.Model(inputs=input, outputs=encoded)

    adam = Adam(learning_rate=0.001)
    autoencoder.compile(optimizer=adam, loss='mse')
    autoencoder.fit(x_train, x_train, epochs=500, batch_size=64, shuffle=True)
    disease_encodeds = encoder.predict(x_train)
    return disease_encodeds

def dis_auto_encoder3(x_train):
    input = layers.Input(256)
    encoded = layers.Dense(64, activation='elu')(input)
    decoded = layers.Dense(256, activation='elu')(encoded)
    autoencoder = models.Model(inputs=input, outputs=decoded)
    encoder = models.Model(inputs=input, outputs=encoded)

    adam = Adam(learning_rate=0.001)
    autoencoder.compile(optimizer=adam, loss='mse')
    autoencoder.fit(x_train, x_train, epochs=500, batch_size=64, shuffle=True)
    disease_encodeds = encoder.predict(x_train)
    return disease_encodeds

def tripletNET_1(args):
    in1 = keras_layers.Input(args.midim)
    in2 = keras_layers.Input(args.didim)
    in3 = keras_layers.Input(args.didim)

    f_mi = miNET(args)
    f_d1 = diNET(args)
    f_d2 = diNET(args) 

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

# %%
def xuly(args,triplets,fold, loop_i):
    # (1)
    if (args.fi_feature == '../IN/Q18.3/kfold/') or (args.fi_feature == '../IN/Q18.3_HMDD3/kfold/'):
        misim_data = pd.read_csv(args.fi_feature + 'L' + str(loop_i) + '/SR_FS' + str(fold+1) + '.csv', header=None)
        disim_data = pd.read_csv(args.fi_feature + 'L' + str(loop_i) + '/SD_SS' + str(fold+1) + '.csv', header=None)
    else:
        misim_data = pd.read_csv(args.fi_feature + 'L' + str(loop_i) + '/SM_fold' + str(fold) + '.csv', header=None)
        disim_data = pd.read_csv(args.fi_feature + 'L' + str(loop_i) + '/SD_fold' + str(fold) + '.csv', header=None)

    ##################################
    miRNA_embeddings = miRNA_auto_encoder(misim_data)  # Dữ liệu miRNA
    miRNA_embeddings = miRNA_auto_encoder2(miRNA_embeddings)
    miRNA_embeddings = miRNA_auto_encoder3(miRNA_embeddings)

    disease_embeddings = dis_auto_encoder(disim_data)  # Dữ liệu disease
    disease_embeddings = dis_auto_encoder2(disease_embeddings)
    disease_embeddings = dis_auto_encoder3(disease_embeddings)
    ##################################

    idx = np.arange(len(triplets))
    np.random.seed(2022) 
    np.random.shuffle(idx)

    triplets = np.array(triplets)

    tr_mi = misim_data.iloc[triplets[idx, 0]]
    tr_di1 = disim_data.iloc[triplets[idx, 1]]
    tr_di2 = disim_data.iloc[triplets[idx, 2]]
    print('tr_mi.shape, tr_di1.shape, tr_di2.shape',tr_mi.shape, tr_di1.shape, tr_di2.shape)

    tr_miA = miRNA_embeddings[triplets[idx, 0],:]
    tr_di1A = disease_embeddings[triplets[idx, 1],:]
    tr_di2A = disease_embeddings[triplets[idx, 2],:]

    # (2) #Train tripletNET_1
    print("\n--- Train tripletNET_1 ...")
    y = np.array([0] * len(triplets)) 
    tripletnet1 = tripletNET_1(args)
    tripletnet1.compile(loss=triplet_loss1, optimizer='adam')
    _ = tripletnet1.fit([tr_mi, tr_di1, tr_di2, tr_miA, tr_di1A, tr_di2A], y, epochs=args.epochs, verbose=2)
    print("Done")

    # (3) #TEST TRIPLETNET NO USE XGBOOST
    te_triplets, te_y = get_sample_triplet1(args, fold, [3, 4], loop_i) 
    # idx = np.arange(len(te_triplets))
    te_triplets = np.array(te_triplets)
    te_mi = misim_data.iloc[te_triplets[:, 0]]
    te_di1 = disim_data.iloc[te_triplets[:, 1]]
    te_di2 = disim_data.iloc[te_triplets[:, 2]]

    te_miA = miRNA_embeddings[te_triplets[:, 0],:]
    te_di1A = disease_embeddings[te_triplets[:, 1],:]
    te_di2A = disease_embeddings[te_triplets[:, 2],:]

    te_distance = tripletnet1.predict([te_mi, te_di1, te_di2, te_miA, te_di1A, te_di2A])

    anchor = te_distance[:, :args.miemb_size]
    positive = te_distance[:, args.miemb_size:args.miemb_size + args.diemb_size]
    negative = te_distance[:, args.miemb_size + args.diemb_size:]

    # -- Save
    te_X = np.concatenate([anchor, positive], axis=1)
    pickle.dump([te_X, te_y], open(args.fi_out + "Data_test/" + "L" + str(loop_i) + "_From_tripletnet1_fold" + str(fold) + ".pkl", "wb"))

    # (4) #GET TRAIN SET FOR TRADITIONAL
    # print("\n\n--- Lay Train cho traditional, tripletnet1")
    tr_triplets, tr_y = get_sample_triplet1(args, fold, [0, 1], loop_i)
    # print(len(tr_triplets))
    # print(len(tr_y))

    idx = np.arange(len(tr_triplets))
    # print(len(idx)) # khac tren kia
    tr_triplets = np.array(tr_triplets)
    tr_mi = misim_data.iloc[tr_triplets[idx, 0]]
    tr_di1 = disim_data.iloc[tr_triplets[idx, 1]]
    tr_di2 = disim_data.iloc[tr_triplets[idx, 2]]

    tr_miA = miRNA_embeddings[tr_triplets[idx, 0],:]
    tr_di1A = disease_embeddings[tr_triplets[idx, 1],:]
    tr_di2A = disease_embeddings[tr_triplets[idx, 2],:]

    tr_distance = tripletnet1.predict([tr_mi, tr_di1, tr_di2, tr_miA, tr_di1A, tr_di2A])

    tr_anchor = tr_distance[:, :args.miemb_size]
    tr_positive = tr_distance[:, args.miemb_size:args.miemb_size + args.diemb_size] # 128 + 128  = 256 cols

    # --- Save
    tr_X = np.concatenate([tr_anchor, tr_positive], axis=1)
    pickle.dump([tr_X, tr_y], open(args.fi_out + "For combination/" + "L" + str(loop_i) + "_Data_train_from_tripletnet1_fold"+ str(fold) + ".pkl", "wb"))
    return

# %%# %%
def main():
    for loop_i in range (1, args.nloop+1):
        print('Loop ', loop_i)
        for fold in range(args.bgf, args.nfold + args.bgf):
            print('fold ',fold)
            triplets = gen_triplet_idx(args, fold, loop_i)
            xuly(args, triplets, fold, loop_i)
    return

if __name__ == "__main__":
    bg_time = time.time()
    main()
    print("Running time: ", time.time() - bg_time, "s")
