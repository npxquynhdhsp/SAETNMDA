# %%
from params import args
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
# ############################################################
# Get random triplets
# triplet (mi,di1,di2): mi (rows), dis (columns)
# ############################################################
def get_pair(args, ix): #get pair indexes from y_4loai_dis_k
    full_pair = pd.read_csv(args.fi_A + 'y_4loai_dis' + str(ix) +'.csv', header=None)
    pair_trN = np.argwhere(full_pair.values == 0)
    pair_trP = np.argwhere(full_pair.values == 1)
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
def gen_triplet_idx(args, ix):
    # pair_trP, pair_trN, _, _ = get_pair(args, ix)
    pair_trP, pair_trN = get_pair(args, ix)
    np.random.seed(2022)
    triplets = []
    # Scan all miRNAs
    for mir_i in range(args.mi_num): #A.shape[0]
        dis_link_mir_i = pair_trP['dis'][pair_trP['mir'] == mir_i]
        dis_nolink_mir_i = pair_trN['dis'][pair_trN['mir'] == mir_i]
        for dis_link in dis_link_mir_i:
            np.random.shuffle(dis_nolink_mir_i)

            if (args.tile == -1) or (len(dis_nolink_mir_i) < args.tile): # tức lấy hết / ko đủ so với tỉ lệ (cần 30 nhưng k đủ 30).
                args.tile = len(dis_nolink_mir_i)
            for dis_nolink in dis_nolink_mir_i[:args.tile]: #với mỗi mi: với mỗi di liên quan, sinh 30 ko liên quan (nếu tile=30).
                #tile này khác ý nghĩa với các bài khác (Ko phải 30*số lượng dương), vì tương ứng 1 dis, tile = 30 nghĩa là ghép (1-30), có 4344 dương nên len=4344x30.
                triplets.append((mir_i, dis_link, dis_nolink))

    # y = np.array([0] * len(triplets)) # mảng số 0, để khớp code
    # pickle.dump({'triplets': triplets, 'y': y}, 
    pickle.dump({'triplets': triplets}, # save triplets
                open(args.fi_out + "Triplet_sample_train/L1_From_tripletnet1_dis" + str(ix) + ".pkl","wb"))
    print('triplets.shape',len(triplets)) #tile = 30 thì tổng là 30*4344 bộ ba(dương, âm, âm)
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
    input = layers.Input(args.didim)
    encoded = layers.Dense(512, activation='elu')(input)
    decoded = layers.Dense(args.didim, activation='elu')(encoded)
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

def tripletNET_1(args):
    in1 = keras_layers.Input(args.midim)
    in2 = keras_layers.Input(args.didim)
    in3 = keras_layers.Input(args.didim)

    f_mi = miNET(args)
    f_d1 = diNET(args)
    f_d2 = diNET(args) #share

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
def xuly(args,triplets,ix):
    # (1)
    if args.fi_feature == '../IN/Q18.3' + args.temp + '/dis_k/':
        misim_data = pd.read_csv(args.fi_feature + 'SR_FS_dis' + str(ix) + '.csv', header=None)
        disim_data = pd.read_csv(args.fi_feature + 'SD_SS_dis' + str(ix) + '.csv', header=None)
    else: # CHƯA COPY
        misim_data = pd.read_csv(args.fi_feature + 'SM_dis' + str(ix) + '.csv', header=None)
        disim_data = pd.read_csv(args.fi_feature + 'SD_dis' + str(ix) + '.csv', header=None)

    ##################################
    miRNA_embeddings = miRNA_auto_encoder(misim_data)  # Dữ liệu miRNA
    miRNA_embeddings = miRNA_auto_encoder2(miRNA_embeddings)
    miRNA_embeddings = miRNA_auto_encoder3(miRNA_embeddings)

    disease_embeddings = dis_auto_encoder(disim_data)  # Dữ liệu disease
    disease_embeddings = dis_auto_encoder2(disease_embeddings)
    disease_embeddings = dis_auto_encoder3(disease_embeddings)
    ##################################

    idx = np.arange(len(triplets))
    np.random.seed(2022) # Xáo thứ tự thôi, nội dung 3 giá trị chỉ số vẫn đi kèm nhau đúng.
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
    y = np.array([0] * len(triplets))  # mảng số 0, để khớp code
    tripletnet1 = tripletNET_1(args)
    tripletnet1.compile(loss=triplet_loss1, optimizer='adam')
    _ = tripletnet1.fit([tr_mi, tr_di1, tr_di2, tr_miA, tr_di1A, tr_di2A], y, epochs=args.epochs, verbose=2)
    print("Done")

    # (3) #GET TEST TRIPLETNET + PREDICT NO USE XGBOOST
    te_triplets, te_y = get_sample_triplet1(args, ix, [3, 4],loop_i=1)  # te_y lộn xộn 1 0
    # idx = np.arange(len(te_triplets)) # ko dùng, nếu dùng thì xáo test làm khi kethop sẽ ko khớp
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
    pickle.dump([te_X, te_y], open(args.fi_out + "Data_test/L1_From_tripletnet1_dis" + str(ix) + ".pkl", "wb"))

    # (4) #GET TRAIN SET FOR TRADITIONAL
    # print("\n\n--- Lay Train cho traditional, tripletnet1")
    tr_triplets, tr_y = get_sample_triplet1(args, ix, [0, 1], loop_i=1)
    # print(len(tr_triplets))
    # print(len(tr_y))

    idx = np.arange(len(tr_triplets))
    tr_triplets = np.array(tr_triplets)
    tr_mi = misim_data.iloc[tr_triplets[idx, 0]]
    tr_di1 = disim_data.iloc[tr_triplets[idx, 1]]
    tr_di2 = disim_data.iloc[tr_triplets[idx, 2]]

    tr_miA = miRNA_embeddings[tr_triplets[idx, 0],:]
    tr_di1A = disease_embeddings[tr_triplets[idx, 1],:]
    tr_di2A = disease_embeddings[tr_triplets[idx, 2],:]

    tr_distance = tripletnet1.predict([tr_mi, tr_di1, tr_di2, tr_miA, tr_di1A, tr_di2A])

    tr_anchor = tr_distance[:, :args.miemb_size]
    tr_positive = tr_distance[:, args.miemb_size:args.miemb_size + args.diemb_size]

    # --- Save
    tr_X = np.concatenate([tr_anchor, tr_positive], axis=1)
    pickle.dump([tr_X, tr_y], open(args.fi_out + "For combination/L1_Data_train_from_tripletnet1_dis"+ str(ix) + ".pkl", "wb"))
    return

# %%# %%
def main():
    for idx in range(len(args.set_dis)):
        dis_k = args.set_dis[idx]
        print('dis_k ',dis_k)
        triplets = gen_triplet_idx(args, dis_k)
        xuly(args, triplets, dis_k)
    return

if __name__ == "__main__":
    main()
