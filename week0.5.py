import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
# 混淆矩阵和ROC曲线
def readData(path):
    with open(path, 'r') as f:
        line = f.readlines()
    text = line[0].split(',')
    op = []
    for t in text:
        if t != '':
            op.append(float(t))
    return op

def genConMatrix(label_list, data_list, threshold):
    '''generate confusion matrix with particular threshold
    Arg:
        label_list\data_list list
        threshold float
    Return:
        confusion matrix

        Pred\L      Positive    Negative
        True        TP          FP
        False       FN          TN
    '''
    conMatrix = np.zeros([2,2], dtype='uint8')
    for label, data in zip(label_list, data_list):
        pred = data >= threshold
        label = bool(label)
        if label:
            if pred:
                conMatrix[0,0] += 1
            else:
                conMatrix[1,0] += 1
        else:
            if pred:
                conMatrix[0,1] += 1
            else:
                conMatrix[1,1] += 1
    return conMatrix

def calFPR(conMatrix):
    return conMatrix.item(0,1) / (conMatrix.item(0,1) + conMatrix.item(1,1))

def calTPR(conMatrix):
    return conMatrix.item(0,0) / (conMatrix.item(0,0) + conMatrix.item(1,0))



if __name__ == '__main__':
    label_path = 'dataset/week0.5/label.txt'
    data_path = 'dataset/week0.5/data.txt'
    label_list = readData(label_path)
    data_list = readData(data_path)
    max_ = np.max(data_list)
    min_ = np.min(data_list)

    step = 9
    threshold_list = np.arange(min_, max_, (max_ - min_) / step).tolist()
    fpr_list = []
    tpr_list = []
    iter_ = tqdm(threshold_list, total=len(threshold_list))
    for thre in iter_:
        conMatrix = genConMatrix(label_list, data_list, thre)
        fpr = calFPR(conMatrix)
        tpr = calTPR(conMatrix)
        iter_.set_description(f'FTR {fpr:.3}\tTPR {tpr:.3}')
        fpr_list.append(fpr)
        tpr_list.append(tpr)
    # sort in the rising of fpr
    # z = list(zip(fpr_list, tpr_list))
    # z.sort(key=lambda elem: elem[0])
    # fpr_list, tpr_list = zip(*z)
    plt.plot(fpr_list, tpr_list)
    x = y = np.arange(0, 1.1, 0.1).tolist()
    plt.plot(x, y, 'b--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    path = './output/week0.5'
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(os.path.join(path, 'ROC_curve.jpg'))
    plt.show()
