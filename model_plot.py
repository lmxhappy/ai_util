#! /usr/bin/env python3
# -*- coding:utf-8 -*-
'''
模块功能描述：

@author Liu Mingxing
@date 2019/12/5
'''
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np

def buildROC(target_test,test_preds):
    # ****** 这是重点，roc、auc
    fpr, tpr, threshold = metrics.roc_curve(target_test, test_preds)
    roc_auc = metrics.auc(fpr, tpr)

    # title
    plt.title('Receiver Operating Characteristic')

    # 依次画出来fpr和tpr
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc, linestyle='dashed')

    # legend
    plt.legend(loc = 'lower right')

    # 画出来中间那条对角线
    plt.plot([0, 1], [0, 1],'r--')

    # 两个label
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    # 保存图片
    plt.gcf().savefig('roc.png')

if __name__ == '__main__':
    target_test = np.array([0, 0, 1, 1, 0, 1])
    test_preds = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.9])

    buildROC(target_test, test_preds)