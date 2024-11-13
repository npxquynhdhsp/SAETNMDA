# %%
import pickle
import time
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import auc, roc_auc_score, accuracy_score, precision_recall_curve, precision_score
from params import args
from GENERAL_UTILS.Q_eval import eval
from GENERAL_UTILS.Genrl_utils import save_file

#####################################
methods = 'X'  # 'MRX' # M(MLP), X(XGBoost), R(RF), L(Linear regresion)
if args.type_eval == 'kfold':
    loai_tam = '_fold'
    b_ix, e_ix = args.bgf, args.nfold + args.bgf
else:
    if args.type_eval == 'dis_k':
        loai_tam = '_dis'
    else:
        loai_tam = '_gl'
    b_ix, e_ix = 0, len(args.set_dis)


###########################################################################

# %%
def read_data(args, ix, triplet_i, loop_i):
    print("--- READ TRAIN SET ---")
    if args.dis_k_koxoa == '':
        tr_X, tr_y = pickle.load(open(
            args.fi_out + "For combination/L" + str(loop_i) + "_Data_train_from_tripletnet" + str(
                triplet_i) + args.dis_k_koxoa + loai_tam + str(ix) + ".pkl", "rb"))
    else:
        tr_X, tr_y = pickle.load(
            open(args.fi_out + "For combination/L" + str(loop_i) + "_Data_train_from_tripletnet" + str(
                triplet_i) + args.dis_k_koxoa + loai_tam + ".pkl", "rb"))
    tr_y = np.array(tr_y)
    print('tr_X.shape', np.array(tr_X).shape)
    print(tr_y.shape)

    print("--- READ TEST SET ---")
    te_X, te_y = pickle.load(open(args.fi_out + "Data_test/L" + str(loop_i) + "_From_tripletnet" + str(
        triplet_i) + args.dis_k_koxoa + loai_tam + str(ix) + ".pkl", "rb"))
    te_y = np.array(te_y)
    print('te_X.shape', np.array(te_X).shape)
    print(te_y.shape)
    return tr_X, tr_y, te_X, te_y


def get_yprob_ypred(tr_X, tr_y, te_X, te_y, args, ix, method, triplet_i, loop_i):
    if method == 'M':  # MLR
        from sklearn.neural_network import MLPRegressor
        model = MLPRegressor(hidden_layer_sizes=(20, 20, 20))
        model.fit(tr_X, tr_y)
        prob_y = model.predict(te_X)
        pred_y = np.array(prob_y > 0)
    elif method == 'L':  # Linear
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(tr_X, tr_y)
        prob_y = model.predict(te_X)
        pred_y = np.array(prob_y > 0)
    elif method == 'X':  # xgboost
        from xgboost import XGBClassifier
        # code estimator from Q16_6
        lrr = args.lrr  # Q
        ne = args.xgne  # ori 500
        model = XGBClassifier(booster='gbtree', n_jobs=2, learning_rate=lrr, n_estimators=ne, random_state=48)
        model.fit(tr_X, tr_y)
        prob_y = model.predict_proba(te_X)[:, 1]
        pred_y = model.predict(te_X)
    else:  # RF
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=500, max_depth=30, random_state=0)
        model.fit(tr_X, tr_y)
        prob_y = model.predict_proba(te_X)[:, 1]
        pred_y = model.predict(te_X)

    np.savetxt(args.fi_out + 'Results/Combination/L' + str(loop_i) + '_prob_' + method + '_triplet' + str(
        triplet_i) + args.dis_k_koxoa + \
               loai_tam + str(ix) + '.csv', prob_y)
    # np.savetxt(args.fi_out + 'Results/Combination/Loop' + str(loop_i) + '_pred_' + method + '_triplet' + str(triplet_i) + args.dis_k_koxoa +\
    #            loai_tam + str(ix) + '.csv', pred_y, fmt='%d')
    np.savetxt(args.fi_out + 'Results/Combination/L' + str(loop_i) + '_true_' + method + '_triplet' + str(
        triplet_i) + args.dis_k_koxoa + \
               loai_tam + str(ix) + '.csv', te_y, fmt="%d")
    return prob_y, pred_y


def get_test_score(yprob, ypred, ytrue):
    acc = accuracy_score(ytrue, ypred)
    pre = precision_score(ytrue, ypred)
    auc_ = roc_auc_score(ytrue, yprob)  # ko đặt trùng tên auc, mà phải auc_
    precision, recall, _ = precision_recall_curve(ytrue, yprob)
    aupr_ = auc(recall, precision)
    return acc, pre, auc_, aupr_


# %%
def main(args, loop_i):  # prob of train
    for itam in range(b_ix, e_ix):
        if args.type_eval == 'kfold':
            ix = itam
            print('########################Fold ', ix, '######################')
        else:
            ix = args.set_dis[itam]
            print('########################Dis: ', ix, '######################')
        print('Triplet 1')
        tr_X_1, tr_y_1, te_X_1, te_y_1 = read_data(args, ix, 1, loop_i)
        prob_y_trp1_m, pred_y_trp1_m = get_yprob_ypred(tr_X_1, tr_y_1, te_X_1, te_y_1, \
                                                       args, ix, method, 1, loop_i)  # test

        print('Triplet 2')
        tr_X_2, tr_y_2, te_X_2, te_y_2 = read_data(args, ix, 2, loop_i)
        prob_y_trp2_m, pred_y_trp2_m = get_yprob_ypred(tr_X_2, tr_y_2, te_X_2, te_y_2, \
                                                       args, ix, method, 2, loop_i)
    return

def main4(args, loop_i, method):# thêm trọng số nghịch đảo
    ACC, PRE, AUC, AUPR = [], [], [], []  # each loop
    name_trb_true = 'L' + str(loop_i) + '_true_trb' + method + args.dis_k_koxoa
    name_trb_prob = 'L' + str(loop_i) + '_prob_trb' + method + args.dis_k_koxoa
    for itam in range(b_ix, e_ix):
        if args.type_eval == 'kfold':
            ix = itam
            print('########################Fold ', ix, '######################')
        else:
            ix = args.set_dis[itam]
            print('########################Dis: ', ix, '######################')
        print('Triplet 1')
        tr_X_1, tr_y_1, te_X_1, te_y_1 = read_data(args, ix, 1, loop_i)
        prob_y_trp1_m = np.genfromtxt(args.fi_out + 'Results/Combination/L' + str(loop_i) + '_prob_' + \
                                      method + '_triplet1' + args.dis_k_koxoa + loai_tam + str(ix) + '.csv')

        print('Triplet 2')
        tr_X_2, tr_y_2, te_X_2, te_y_2 = read_data(args, ix, 2, loop_i)
        prob_y_trp2_m = np.genfromtxt(args.fi_out + 'Results/Combination/L' + str(loop_i) + '_prob_' + \
                                      method + '_triplet2' + args.dis_k_koxoa + loai_tam + str(ix) + '.csv')

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

        save_file(args.fi_out + 'Results/Combination/', name_trb_true, loai_tam, ix, 'onedim_int', te_y_1)  # q
        save_file(args.fi_out + 'Results/Combination/', name_trb_prob, loai_tam, ix, 'onedim', prob_y_m)  # q

        acc_m, pre_m, auc_m, aupr_m = get_test_score(prob_y_m, pred_y_m, te_y_1)
        print('AUC, AUPR', auc_m, aupr_m)

        ACC.append(acc_m)
        PRE.append(pre_m)
        AUC.append(auc_m)
        AUPR.append(aupr_m)

    return AUC, AUPR

# %%
if __name__ == "__main__":
    bg_time = time.time()
    for i in range(len(methods)):
        print('*' * 100)
        method = methods[i]
        print('method:', method)
        print('-' *50)

        AUC_all, AUPR_all = [], []
        for loop_i in range(1, args.nloop + 1):
            print('Loop ', loop_i)
            main(args, loop_i)
            AUC_l, AUPR_l = main4(args, loop_i, method)
            AUC_all.append(AUC_l)
            AUPR_all.append(AUPR_l)

        print('FINAL RESULT: ', np.mean(AUC_all), np.mean(AUPR_all))



