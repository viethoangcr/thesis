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
    
    def __union_one_half_set(self, samples, new_index):
        index = np.array([],dtype=np.int64)
        N = len(samples)
        j = 0
        for i in range(N):
            if i % 2 == 0:
                sample = samples[i]
                ind = new_index[j]
                j = j + 1
                index = np.union1d(index, sample[ind])
        return index.tolist()
        
    def __union_one_third_set(self, samples, new_index):
        index = np.array([],dtype=np.int64)
        N = len(samples)
        j = 0
        for i in range(N):
            if i % 3 == 0:
                sample = samples[i]
                ind = new_index[j]
                j = j + 1
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
        
    def train(self, X_init,y_init, beta=0.01, g=1, debug=False):
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
            for sample in subsamples:
                try:
                    index.append(self.__get_support_vectors(X[sample,],y[sample,]))
                except BaseException as error:
                    print(error)
                    return None
                    
            new_X_index = self.__union_set(subsamples,index)
            
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
        
    def train_one_third(self, X_init,y_init, beta=0.01, g=1, debug=True):
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
            error_index = []
            
            for sample in subsamples:
                #try:
                iSample = (iSample + 1) % 3
                if iSample == 1:
                    trainSVC = self.__get_svc(X[sample,], y[sample,])
                    index.append(trainSVC.support_)
                else:
                    for iTestSample in sample:
                        expected_X = trainSVC.predict(X[iTestSample,])
                        if expected_X != y[iTestSample,]:
                            error_index.append(iTestSample)
                #except BaseException as error:
                #    print(error)
                #    return Nones
                    
            new_X_index = self.__union_one_third_set(subsamples, index)
            new_X_index = np.union1d(new_X_index, error_index)
            new_X_index = [int(v) for v in new_X_index]
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
    
    def train_one_half(self, X_init,y_init, beta=0.01, g=1, debug=True):
        c = 1
        i = 0
        X = X_init
        y = y_init
        N = X.shape[0]
        n = [N]
        flag = 0
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
            error_index = []
            
            for sample in subsamples:
                #try:
                if (flag == 1)
                    trainSVC = self.__get_svc(X[sample,], y[sample,])
                    index.append(trainSVC.support_)
                    continue
                
                iSample = 1 - iSample
                if iSample == 1:
                    trainSVC = self.__get_svc(X[sample,], y[sample,])
                    index.append(trainSVC.support_)
                else:
                    for iTestSample in sample:
                        expected_X = trainSVC.predict(X[iTestSample,])
                        if expected_X != y[iTestSample,]:
                            error_index.append(iTestSample)
                #except BaseException as error:
                #    print(error)
                #    return Nones
                    
            if flag == 0
                new_X_index = self.__union_one_half_set(subsamples, index)
                new_X_index = np.union1d(new_X_index, error_index)
            else
                new_X_index = self.__union_set(subsamples, index)
                flag = 2

            new_X_index = [int(v) for v in new_X_index]

            X = X[new_X_index,]
            y = y[new_X_index,]
            n.append(X.shape[0])
    
            if debug:
                print("Number of SVs: %d / %d" % (n[i], n[i-1]))
                print("Execute time (in second): %s" % (time.time() - starttime))

            if flag == 2:
                break
            
            if  g*n[i]*k/c >= (n[i-1]-n[i])**2:
                flag = 1
        svc = SVC(**self.svm_parameters)
        self.model = svc.fit(X,y)
        return self.model
    
    def train_by_file(self, svmlight_file_address:str, beta=0.01, g=1, debug=False):
        X_train, y_train = datasets.load_svmlight_file(svmlight_file_address)
        return self.train_one_third(X_train, y_train, beta, g, debug)
        
    def train_large_file(self, svmlight_file_address:str, beta=0.01, g=1, debug=False, temp_folder=None):
        c = 1
        i = 0
        handler = SvmlightFileHandler(svmlight_file_address)
        N = handler.size
        n = [N]
        starttime = time.time()
        
        while True:
            
            if debug == 1:
                print()
                print("Iteration " + str(i+1))
                
            i = i + 1
            k = math.ceil(i*beta*N)
            m = math.ceil(n[i-1] * g / k)
            subsamples = self.__create_subsamples(n[i-1], m, k)
            index = []
            for sample in subsamples:
                try:
                    (X_sample, y_sample) = handler[sample]
                    index.append(self.__get_support_vectors(X_sample, y_sample))
                except BaseException as error:
                    global gsample, hdl
                    hdl = handler
                    gsample = sample
                    print(error)
                    return None
                    
            new_X_index = self.__union_set(subsamples,index)
            
            handler.filter(new_X_index)
            n.append(handler.size)
    
            if debug:
                print("Number of SVs: %d / %d" % (n[i], n[i-1]))
                print("Execute time (in second): %s" % (time.time() - starttime))
            
            if  g*n[i]*k/c >= (n[i-1]-n[i])**2:
                break
            
        svc = SVC(**self.svm_parameters)
        (X,y) = handler[0:handler.size]
        self.model = svc.fit(X,y)
        return self.model

def main():
    start_time = time.time()
    svm_para = {'C': 10.0, 'kernel': 'rbf', 'gamma': 1.667, 'verbose': False}
#    svm_para = {'kernel': 'linear', 'verbose': False}
    # loading data
#    X_train, y_train = datasets.load_svmlight_file(r'./../dataset/mnist_train_576_rbf_8vr.dat')
#    X_test, y_test = datasets.load_svmlight_file(r'./../dataset/mnist_test_576_rbf_8vr.dat')
    
    
    svm_para = {'C': 10.0, 'kernel': 'rbf', 'gamma': 0.00002, 'tol': 0.01, 'verbose': False}

    # test ramdom sampling
    RS_SVM = RandomSamplingSVM(svm_para)
    model = RS_SVM.train_large_file(r'./../dataset/covtype_tr_2vr.data', debug=True)
    
    
    print("--- %s seconds ---" % (time.time() - start_time))
    
    if model is None:
        print("Can not train the dataset")
    else:
        
        X_test, y_test = datasets.load_svmlight_file(r'./../dataset/covtype_tst_2vr.data')
        ratio = model.score(X_test,y_test)
        print(ratio)
        print("--- %s seconds ---" % (time.time() - start_time))
        
if __name__ == '__main__':
    main()
