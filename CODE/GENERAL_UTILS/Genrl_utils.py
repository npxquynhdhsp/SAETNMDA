#%%
import numpy as np

#%%
def save_file(path,name,loai_tam,ix, loai_label, value):

    if loai_label == 'onedim_int':  # mảng 1 chiều, nguyên
        np.savetxt(path + name + loai_tam + str(ix) + '.csv', value, fmt='%d')
    else:
        if loai_label == 'onedim':  # mảng 1 chiều, bt
            np.savetxt(path + name + loai_tam + str(ix) + '.csv', value)
        else:
            if loai_label == 'twodim_int':  # mảng 2 chiều, nguyên
                np.savetxt(path + name + loai_tam + str(ix) + '.csv', value, delimiter=',', fmt='%d')
            else:  # mảng 2 chiều, lẻ
                np.savetxt(path + name + loai_tam + str(ix) + '.csv', value, delimiter=',')
    return

# %%
def rutgon_sub(X, y):
    X_pos = X[y == 1]
    X_neg = X[y == 0]
    num_pos = X_pos.shape[0]
    X2_neg = X_neg[0:num_pos]
    X2 = np.vstack((X_pos, X2_neg))
    y2 = np.hstack((np.ones(shape=(num_pos)), np.zeros(shape=(num_pos)))).reshape(-1)
    return X2, y2

def rutgon(tr_X, tr_y, te_X, te_y):
    tr2_X, tr2_y = rutgon_sub(tr_X, tr_y)
    te2_X, te2_y = rutgon_sub(te_X, te_y)
    # print('RUT GON TRAIN TEST:')
    # print(tr2_X.shape, tr2_y.shape, te2_X.shape, te2_y.shape)
    return tr2_X, tr2_y, te2_X, te2_y

# %%
def sosanh(path1, f1, path2, f2):
    import numpy as np
    a1 = np.genfromtxt(path1 + f1,delimiter=',')
    a2 = np.genfromtxt(path2 + f2,delimiter=',')
    print(np.argwhere(a1!=a2))
    return


# %%