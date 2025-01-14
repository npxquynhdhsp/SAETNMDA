# %%
import argparse
import numpy as np
############CHOICE PARAMETERS#################
# (1) Database ('HMDD2' or 'HMDD3')
db = 'HMDD2'
# (2) Type of eval ('kfold' or 'dis_k')
type_eval = 'kfold'
##############################################
# (3) TYPE OF KERNEL =======================
dir_temp = 'Q18.3/'
# dir_temp = 'A from Q18,kernel goc(binh thuong)/'
# (4) ko xoa dis_k =======================
dis_k_koxoa = ''

# %%
nloop = 1
nfold = 5
bgf = 0 # begin fold = 0 or 1 của SD,SR, (Q18 default bgf = 0)
if db == 'HMDD2':
    temp = ''
    set_dis = [50, 59, 236]
    if type_eval == 'dis_k_gl':
        set_dis = np.arange(1,5431)  # gl
    tile = 30  # ti le mau am, duong trong cac triplet, tile = -1 la lay het. Ko phai ti le am duong nhu bt.
else:
    temp = '_' + db
    set_dis = [3, 5, 9]
    tile = 15 # Q 50
if type_eval == 'dis_k':
    nloop = 1
    # dis_k_koxoa = '_koxoa'

# %%
cb = 1 # = 1: balanced training

ne = 100 # number of epoch of triplet, org 100
alpha = 0.2
miemb_size, diemb_size = 128, 128
batch_size = 128 # hong thay dung

lrr = 0.2 # Q
xgne = 500 # XGboost, ori 500

if db == 'HMDD2':
    di_num, mi_num = 383, 495
    didim, midim = 383, 495
else:
    di_num, mi_num = 374, 788
    didim, midim = 374, 788


if dir_temp == 'A from Q18,kernel goc(binh thuong)/':
    fi_feature = '../IN/KERNEL GOC/' + type_eval + '/' + dir_temp
else:
    fi_feature = '../IN/Q18.3' + temp + '/' + type_eval + '/'
fi_A = '../IN/Q18.3' + temp + '/' + type_eval + '/'
fi_out = './OUT Q_' + db + '/' + type_eval + '/' + dir_temp
print('dir_temp',dir_temp)
print('fi_feature',fi_feature)

# %%
def parameter_parser():
    parser = argparse.ArgumentParser(description="Q23_TripletNetwork_MDA.")
    parser.add_argument("--fi_feature",
                        nargs="?",
                        default=fi_feature,
                        help="fi_feature.")
    parser.add_argument("--fi_A",
                        nargs="?",
                        default=fi_A,
                        help="y_train")
    parser.add_argument("--fi_out",
                        nargs="?",
                        default=fi_out,
                        help="fi_out.")
    parser.add_argument("--db",
                       default=db)
    parser.add_argument("--type_eval",
                        nargs="?",
                        default=type_eval,
                        help="kfold/dis_k.")
    parser.add_argument("--temp",
                        default=temp)
    parser.add_argument("--dis_k_koxoa",
                        default=dis_k_koxoa)
    parser.add_argument("--epochs",
                    type=int,
                    default=ne,
                    help="Number of training epochs of triplet.")
    parser.add_argument("--lrr",
                    default=lrr,
                    help="learning rate of XGBoost.")
    parser.add_argument("--xgne",
                    default=xgne,
                    help="Number of training epochs of XGBoost.")
    parser.add_argument("--nloop",
                        type=int,
                        default=nloop,
                        help="Number of loop.")
    parser.add_argument("--nfold",
                        type=int,
                        default=nfold,
                        help="n cross-validation.")
    parser.add_argument("--bgf",
                        type=int,
                        default=bgf,
                        help="begin number of fold, 0 or 1.")
    parser.add_argument("--set_dis",
                        default=set_dis,
                        help="[3,5,9].")

    ##############################
    parser.add_argument("--cb",
                        type=int,
                        default=cb)
    parser.add_argument("--tile",
                        type=int,
                        default=tile)
    parser.add_argument("--alpha",
                        type=float,
                        default=alpha)
    parser.add_argument("--miemb_size",
                        type=int,
                        default=miemb_size)
    parser.add_argument("--diemb_size",
                        type=int,
                        default=diemb_size)
    parser.add_argument("--mi_num",
                        type=int,
                        default=mi_num)
    parser.add_argument("--di_num",
                        type=int,
                        default=di_num)
    parser.add_argument("--midim",
                        type=int,
                        default=midim)
    parser.add_argument("--didim",
                        type=int,
                        default=didim)
    parser.add_argument("--batch_size",
                        type=int,
                        default=batch_size)

    ##############################
    return parser.parse_args()
args = parameter_parser()