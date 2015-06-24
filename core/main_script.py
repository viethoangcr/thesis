# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 00:51:07 2015

@author: viethoangcr
"""

from RandomSamplingSVM import RandomSamplingSVM
import time
from sklearn import datasets
from sklearn import svm


def single_svm(train_file, test_file, kernel):
    print("Single SVM")
    start_time = time.time()
    X_train, y_train = datasets.load_svmlight_file(train_file)
    svc = svm.SVC(**kernel)
    svc.fit(X_train,y_train)
    
    print("Remain SVs: " + str(svc.n_support_))
    print("Training time: %s" % (time.time() - start_time))
    X_test, y_test = datasets.load_svmlight_file(test_file)
    ratio = svc.score(X_test,y_test)
    print("Accuracy %f" % ratio)
    print("Total time: %s" % (time.time() - start_time))
    
def rs_svm(train_file, test_file, kernel):
    print("Ramdom Sampling SVM")
    start_time = time.time()
    RS_SVM = RandomSamplingSVM(kernel)
    model = RS_SVM.train_by_file(train_file, debug=True)
    print("Remain SVs: " + str(model.n_support_))
    print("Training time: %s" % (time.time() - start_time))
    X_test, y_test = datasets.load_svmlight_file(test_file)
    ratio = model.score(X_test,y_test)
    print("Accuracy %f" % ratio)
    print("Total time: %s" % (time.time() - start_time))

def rs_svm_ratio_test(train_file, test_file, kernel):
    print("Ramdom Sampling SVM with Ratio")
    xTrain, yTrain = datasets.load_svmlight_file(svmlight_file_address)
    xTest, yTest = datasets.load_svmlight_file(test_file)

    for iRatio in range(9):
        RS_SVM = RandomSamplingSVM(kernel)

        start_time = time.time()
        trainRatio = iRatio * 0.1 + 0.2
        
        model = trainWithRatio(trainRatio, xTrain, yTrain)

        print("Remain SVs: " + str(model.n_support_))
        print("Training time: %s" % (time.time() - start_time))
        
        testRatio = model.score(xTest, yTest)

        print("Accuracy %f" % testRatio)
        print("Total time: %s" % (time.time() - start_time))
        print()
    
def rs_svm_large_file(train_file, test_file, kernel):
    print("Ramdom Sampling SVM for large dataset")
    start_time = time.time()
    RS_SVM = RandomSamplingSVM(kernel)
    model = RS_SVM.train_large_file(train_file, debug=True, temp_folder='./')
    print("Remain SVs: " + str(model.n_support_))
    print("Training time: %s" % (time.time() - start_time))
    X_test, y_test = datasets.load_svmlight_file(test_file)
    ratio = model.score(X_test,y_test)
    print("Accuracy %f" % ratio)
    print("Total time: %s" % (time.time() - start_time))

def test(train_file, test_file, kernel):
    print("DATASET: " + train_file)
    
    # Single
#    single_svm(train_file, test_file, kernel)
    
    # RS_SVM using RAM
    #rs_svm(train_file, test_file, kernel)
    rs_svm_ratio_test(train_file, test_file, kernel)

    
    #RS_SVM using disk as cache
    #rs_svm_large_file(train_file, test_file, kernel)

#isequal = lambda x, y : True if (x-y).nnz == 0 else False
#
#X_train, y_train = datasets.load_svmlight_file(r'./w8a')
#handler = SvmlightFileHandler(r'./w8a')

#start_time = time.time()

svm_para = {'C': 10.0, 'kernel': 'rbf', 'gamma': 1.667, 'verbose': False}
test(r'./dataset/mnist_train_576_rbf_8vr.dat', r'./dataset/mnist_test_576_rbf_8vr.dat', svm_para)

#test(r'./../dataset/mnist_train_784_poly_8vr.dat', r'./../dataset/mnist_test_784_poly_8vr.dat', svm_para)

#svm_para = {'C': 10.0, 'kernel': 'rbf', 'gamma': 0.00002, 'tol': 0.01, 'verbose': False}
#test(r'./dataset/covtype_tr_2vr.data', r'./dataset/covtype_tst_2vr.data', svm_para)