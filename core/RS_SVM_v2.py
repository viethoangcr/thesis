# -*- coding: utf-8 -*-
"""
Created on Sun May 17 18:03:40 2015

@author: Viet Hoang
"""
# import libraries
import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
import math
from SvmlightFileHandler import SvmlightFileHandler
import time


class RandomSamplingSVM(object):
    def __init__(self, svm_parameters={}):
        self.svm_parameters = svm_parameters
        self.model = None

    def set_svm_params(self, svm_parameters={}):
        self.svm_parameters = svm_parameters

    def __union_set(self, samples, new_index):
        index = np.array([],dtype=np.int64)
        N = len(samples)
        for i in range(N):
            sample = samples[i]
            ind = new_index[i]
            index = np.union1d(index, sample[ind])
        return index.tolist()


    # this function create subsamples from a set
    def __create_subsamples(self, N, m, k):
        ind = []
        x = 0
        P = np.random.permutation(N)
        for i in range(m):
            sub = np.array([])
            if x+k < N:
                sub = P[x:(x+k)]
                x = x + k
            else:
                sub = P[x:N]
                P = np.random.permutation(N)
                sub = np.append(sub,P[0:k - N + x])
                x = k - N + x
            ind.append(sub)
        return ind

    # this function get list of support vectors
    def __get_svc(self, X, y):
        svc = SVC(**self.svm_parameters)
        svc.fit(X,y)
        return svc

    def __get_support_vectors(self, X,y):
        #global DEBUG
        svc = SVC(**self.svm_parameters)
        svc.fit(X,y)
        return svc.support_

    def __select_randomly(self, n:int, k:int, in_order:bool=False):
        """
        This function randomly choose k items from total n items

        """
        result = None
        if n >= k and k>= 0:
            result = np.random.permutation(n)[0:k]
        else:
            result = np.array([], dtype=np.int64)
        if in_order:
            result.sort()
        return result

    def train_one_half(self, X_init,y_init, beta=0.01, g=1, debug=True):
        c = 1
        i = 0
        X = X_init
        y = y_init
        N = X.shape[0]
        n = [N]
        starttime = time.time()

        while True:

            if debug:
                print()
                print("Iteration " + str(i+1))

            i = i + 1
            k = math.ceil(i*beta*N)
            m = math.ceil(n[i-1] * g / k)
            subsamples = self.__create_subsamples(n[i-1], m, k)
            index = []

            iSample = 0
            trainSVC = None

            for sample in subsamples:
                #try:
                iSample = 1 - iSample
                if iSample == 1:
                    trainSVC = self.__get_svc(X[sample,], y[sample,])
                    index.append(trainSVC.support_)
                else:
                    y_predict = trainSVC.predict(X[sample,])
                    index.append(np.nonzero((y_predict - y[sample,]) != 0)[0])

            new_X_index = self.__union_set(subsamples, index)

            X = X[new_X_index,]
            y = y[new_X_index,]
            n.append(X.shape[0])

            if debug:
                print("Number of SVs: %d / %d" % (n[i], n[i-1]))
                print("Execute time (in second): %s" % (time.time() - starttime))

            if  g*n[i]*k/c >= (n[i-1]-n[i])**2:
                break
        svc = SVC(**self.svm_parameters)
        self.model = svc.fit(X,y)
        return self.model

    def train_one_half_v2(self, X_init,y_init, beta=0.01, g=1, debug=True):
        print("train_one_half_v2 by Hoang", flush=True)
        c = 1
        i = 0
        X = X_init
        y = y_init
        N = X.shape[0]
        n = [N]
        starttime = time.time()

        while True:

            if debug:
                print()
                print("Iteration " + str(i+1), flush=True)

            i = i + 1
            k = math.ceil(i*beta*n[-1])
            m = math.ceil(n[i-1] * g / k)
            subsamples = self.__create_subsamples(n[i-1], m, k)
            index = []

            iSample = 0
            trainSVC = None

            for sample in subsamples:
                #try:
                iSample = 1 - iSample
                if iSample == 1:
                    trainSVC = self.__get_svc(X[sample,], y[sample,])
                    index.append(trainSVC.support_)
                else:
                    y_predict = trainSVC.predict(X[sample,])
                    incor = np.unique(np.nonzero((y_predict - y[sample,]) != 0)[0])
                    cor =np.unique(np.nonzero((y_predict - y[sample,]) == 0)[0])
                    # 
                    delta = 0.5
                    # temp = min(cor.shape[0], max(0, (1 + delta) * incor.shape[0] - index[-1].shape[0]))
                    temp = int(delta * cor.shape[0])
                    #print(cor.shape[0], incor.shape[0], trainSVC.support_.shape[0], temp)
                    index.append(np.concatenate((incor, cor[self.__select_randomly(cor.shape[0], temp)])))
                    

            new_X_index = self.__union_set(subsamples, index)

            X = X[new_X_index,]
            y = y[new_X_index,]
            n.append(X.shape[0])

            if debug:
                print("Number of SVs: %d / %d" % (n[i], n[i-1]), flush=True)
                print("Execute time (in second): %s" % (time.time() - starttime), flush=True)

            if  g*n[i]*i*beta*N/c >= (n[i-1]-n[i])**2:
                break
        svc = SVC(**self.svm_parameters)
        self.model = svc.fit(X,y)
        return self.model

def main():
    
    svm_para = {'C': 10.0, 'kernel': 'rbf', 'gamma': 1.667, 'verbose': False}
    #svm_para = {'kernel': 'linear', 'verbose': False}
    #loading data
    X_train, y_train = datasets.load_svmlight_file(r'./dataset/mnist_train_784_poly_8vr.dat')
    #X_train, y_train = datasets.load_svmlight_file(r'./dataset/covtype_tr_2vr.data')


    #svm_para = {'C': 10.0, 'kernel': 'rbf', 'gamma': 0.00002, 'tol': 0.01, 'verbose': False}

    # test ramdom sampling
    RS_SVM = RandomSamplingSVM(svm_para)
    start_time = time.time()
    model = RS_SVM.train_one_half_v2(X_train, y_train)

    print("Remain SVs: " + str(model.n_support_), flush=True)
    print("--- %s seconds ---" % (time.time() - start_time), flush=True)

    if model is None:
        print("Can not train the dataset", flush=True)
    else:

        X_test, y_test = datasets.load_svmlight_file(r'./dataset/mnist_test_784_poly_8vr.dat')
        #X_test, y_test = datasets.load_svmlight_file(r'./dataset/covtype_tst_2vr.data')
        ratio = model.score(X_test,y_test)
        print(ratio)
        print("--- %s seconds ---" % (time.time() - start_time), flush=True)

if __name__ == '__main__':
    main()
