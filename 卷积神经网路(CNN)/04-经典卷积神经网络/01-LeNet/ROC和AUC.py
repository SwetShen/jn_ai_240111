# 机器学习中关于AUC和ROC
# import sklearn
# sklearn.metrics.auc
# sklearn.metrics.roc_curve
# sklearn.metrics.roc_auc_score

import numpy as np
import matplotlib.pyplot as plt

labels = np.random.rand(1000, )
labels[labels >= 0.5] = 1
labels[labels < 0.5] = 0
thresholds = np.arange(0, 11, 1.1) * 0.1

roc_points = []

# ROC 存在的意义就是，证明模型是不是最好的。
for threshold in thresholds:
    # predict = np.array([0.29987381, 0.33596201, 0.18625308, 0.23248181, 0.98431477, 0.5666045,
    #                     0.18711898, 0.44796161, 0.2678888, 0.21940537])
    predict = np.random.rand(1000, )
    predict[predict >= threshold] = 1
    predict[predict < threshold] = 0

    TP = np.sum(labels * predict)
    FP = np.sum((1 - labels) * predict)
    FN = np.sum(labels * (1 - predict))
    TN = np.sum((1 - labels) * (1 - predict))

    print(TP, FP, FN, TN)

    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)

    roc_points.append([TPR, FPR])

roc_points = np.array(roc_points)
plt.plot(roc_points[:, 0], roc_points[:, 1], 'r-')
plt.title("ROC curve")
plt.show()
