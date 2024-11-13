import numpy as np
from sklearn.metrics import (confusion_matrix, 
                            f1_score, 
                            accuracy_score, 
                            recall_score, 
                            precision_score)
import math


def calc_scores_from_true_pred(y_test, y_pred): 
    scores = dict()
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)

    acc = (tp + tn)/(tp + tn + fp + fn)
    pre = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = 2*pre*rec / (pre+rec)
    precision = pre
    recall = rec
    #MCC = (tp*tn-fp*fn) / math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))

    scores['tn'] = tn
    scores['fp'] = fp
    scores['fn'] = fn
    scores['tp'] = tp
    scores['fpr'] = fpr
    scores['fnr'] = fnr
    scores['f1'] = f1
    scores['accuracy'] = acc
    scores['tpr'] = scores['recall'] = scores['sensitivity'] = tpr
    scores['tnr'] = scores['specificity'] = tnr
    scores['precision'] = precision
    #scores['MCC'] = MCC

    return scores


def calc_scores_KFOLD(arr_y_test, arr_y_pred, name_file):
    scores = dict()
    n_folds = len(arr_y_test)

    tpr = fpr = tnr = fnr = 0.000
    f1 = acc = 0.000
    recall = precision = 0.000
    MCC = 0.000

    for y_test, y_pred in zip(arr_y_test, arr_y_pred):
        y_pred = y_pred.reshape(-1)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        tpr += tp / (tp + fn)
        tnr += tn / (tn + fp)
        fpr += fp / (fp + tn)
        fnr += fn / (fn + tp)

        acc += (tp + tn)/(tp + tn + fp + fn)
        pre = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1 += 2*pre*rec / (pre+rec)
        precision += pre
        recall += rec
        #MCC += (tp*tn-fp*fn) / math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))

    f1 = f1 / n_folds
    acc = acc / n_folds
    precision = precision / n_folds
    #MCC = MCC / n_folds
    tpr = tpr / n_folds
    tnr = tnr / n_folds
    fpr = fpr / n_folds
    fnr = fnr / n_folds

    scores['tn'] = tn
    scores['fp'] = fp
    scores['fn'] = fn
    scores['tp'] = tp
    scores['fpr'] = fpr
    scores['fnr'] = fnr
    scores['accuracy'] = acc
    scores['tpr'] = scores['recall'] = scores['sensitivity'] = tpr
    scores['tnr'] = scores['specificity'] = tnr
    scores['precision'] = precision
    scores['f1'] = f1
    #scores['MCC'] = MCC
    
    f=open(name_file,'w')
    f.writelines(['TN', '\t','FP', '\t','FN', '\t','TP', '\t','FPR', '\t','FNR', '\t','Acc', '\t','TPR', '\t','Recall', '\t','Sensitivity', '\t','TNR', '\t','Specificity', '\t','Precision', '\t','F1-score bthuong','\n'])
    f.writelines([str(tn),'\t',str(fp),'\t',str(fn),'\t',str(tp),'\t',              str(fpr),'\t',str(fnr),'\t',str(acc),'\t',         str(tpr),'\t',str(tpr),'\t',str(tpr),'\t',                     str(tnr),'\t',str(tnr),'\t',str(precision),'\t',str(f1),'\n'])
    f.close()
    
    return scores


def calc_scores_KFOLD_cm(fname):
    # Reading arr_cm from file at fname
    arr_cm = np.genfromtxt(fname, delimiter=',')

    scores = dict()
    n_folds = arr_cm.shape[0] // 2

    tpr = fpr = tnr = fnr = 0.0
    f1 = acc = 0.0
    recall = precision = 0.0
    MCC = 0.0

    for i in range(0, arr_cm.shape[0], 2):
        cm = arr_cm[i:i+2]
        # print('cm %d'%(i), cm, sep='\n')
        tn, fp, fn, tp = cm.ravel()
        
        tpr += tp / (tp + fn)
        tnr += tn / (tn + fp)
        fpr += fp / (fp + tn)
        fnr += fn / (fn + tp)

        acc += (tp + tn)/(tp + tn + fp + fn)
        pre = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1 += 2*pre*rec / (pre+rec)
        precision += pre
        recall += rec
        #MCC += (tp*tn-fp*fn) / math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))

    f1 = f1 / n_folds
    acc = acc / n_folds
    precision = precision / n_folds
    #MCC = MCC / n_folds
    tpr = tpr / n_folds
    tnr = tnr / n_folds
    fpr = fpr / n_folds
    fnr = fnr / n_folds

    scores['tn'] = tn
    scores['fp'] = fp
    scores['fn'] = fn
    scores['tp'] = tp
    scores['fpr'] = fpr
    scores['fnr'] = fnr
    scores['f1'] = f1
    scores['accuracy'] = acc
    scores['tpr'] = scores['recall'] = scores['sensitivity'] = tpr
    scores['tnr'] = scores['specificity'] = tnr
    scores['precision'] = precision
    #scores['MCC'] = MCC

    return scores