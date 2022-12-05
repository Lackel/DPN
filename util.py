import numpy as np
from scipy.optimize import linear_sum_assignment

def hungray_aligment(y_true, y_pred):
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D))
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = np.transpose(np.asarray(linear_sum_assignment(w.max() - w)))
    return ind, w

def clustering_accuracy_score(y_true, y_pred, known_lab):
    ind, w = hungray_aligment(y_true, y_pred)
    acc = sum([w[i, j] for i, j in ind]) / y_pred.size
    ind_map = {j: i for i, j in ind}
    
    old_acc = 0
    total_old_instances = 0
    for i in known_lab:
        old_acc += w[ind_map[i], i]
        total_old_instances += sum(w[:, i])
    old_acc /= total_old_instances
    
    new_acc = 0
    total_new_instances = 0
    for i in range(len(np.unique(y_true))):
        if i not in known_lab:
            new_acc += w[ind_map[i], i]
            total_new_instances += sum(w[:, i])
    new_acc /= total_new_instances
    return (round(acc*100, 2), round(old_acc*100, 2), round(new_acc*100, 2))

def clustering_score(y_true, y_pred, known_lab):
    return {'ACC_all': clustering_accuracy_score(y_true, y_pred, known_lab)[0],
            'ACC_known': clustering_accuracy_score(y_true, y_pred, known_lab)[1],
            'ACC_novel': clustering_accuracy_score(y_true, y_pred, known_lab)[2]}


    

