# -*- coding: utf-8 -*-
"""
Created on Sun May 17 18:03:40 2015

@author: Viet Hoang
@author: ultimate.lead
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
    # len of set = [k/ratio, k*ratio, k/ratio, ...]
    # m = the number of samples
    # N = the init set

    def __createRatioIndexData(self, N, m, k, ratio):
        indexSub = []

        x = 0
        P = np.random.permutation(N)

        trainSize = math.ceil(2 * k * ratio)
        testSize = 2 * k - trainSize

        for i in range(m):
            sub = np.array([])

            if i % 2 == 0:
                currentSize = trainSize
            else:
                currentSize = testSize

            if x + currentSize < N:
                sub = P[x : (x + currentSize)]
                x = x + currentSize
            else:
                sub = P[x : N]
                P = np.random.permutation(N)
                # sub = np.append(sub, P[0 : currentSize - N + x])
                # x = currentSize - N + x

            indexSub.append(sub)
        return indexSub

    # this function create subsamples from a set
    def __create_subsamples(self, N, m, k):
        ind = []
        x = 0
        P = np.random.permutation(N)

        for i in range(m):
            sub = np.array([])

            if x + k < N:
                sub = P[x:(x + k)]
                x = x + k
            else:
                sub = P[x:N]
                P = np.random.permutation(N)
                sub = np.append(sub, P[0:k - N + x])
                x = k - N + x

            ind.append(sub)

        return ind

    def __get_svc(self, X, y):
        svc = SVC(**self.svm_parameters)
        svc.fit(X,y)
        return svc

    # this function get list of support vectors
    def __get_support_vectors(self, X,y):
        svc = SVC(**self.svm_parameters)
        svc.fit(X, y)
        return svc.support_

    def train(self, X_init,y_init, beta=0.01, g=1, debug=False):
        print("Original Random Sampling SVM", flush=True)

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
                print("Number of SVs: %d / %d" % (n[i], n[i-1]), flush=True)
                print("Execute time (in second): %s" % (time.time() - starttime), flush=True)

            if  g*n[i]*k/c >= (n[i-1]-n[i])**2:
                break
        svc = SVC(**self.svm_parameters)
        self.model = svc.fit(X,y)
        return self.model

    def train_small(self, X_init,y_init, beta=0.01, g=1, debug=False):
        print("Small Random Sampling SVM", flush=True)

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
            k = math.ceil(i*beta*n[i - 1])
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
                print("Number of SVs: %d / %d" % (n[i], n[i-1]), flush=True)
                print("Execute time (in second): %s" % (time.time() - starttime), flush=True)

            if  g*n[i]*i*beta*N/c >= (n[i-1]-n[i])**2:
                break
        svc = SVC(**self.svm_parameters)
        self.model = svc.fit(X,y)
        return self.model

    # ratio = #train / #total
    def trainWithRatio(self, ratio, xTrain, yTrain, beta=0.01, g=1, debug=True):
        if ratio == 0:
            # Dont train any but test, what model to test? :lol:
            return

        print("Adjusted RS SVM ver 2 with Train/Total = %f" %ratio, flush=True)
        c = 1
        i = 0
        X = xTrain
        Y = yTrain
        N = X.shape[0]
        n = [N]

        startTime = time.time()

        while True:
            i = i + 1
            print("Iteration = %d" %i, flush=True)

            k = math.ceil(i*beta*n[i - 1])
            m = math.ceil(n[i - 1] * g / k)

            trainSize = math.ceil(2 * k * ratio)
            testSize = 2 * k - trainSize

            indexSub = self.__createRatioIndexData(n[i - 1], m, k, ratio)
            print(len(indexSub))

            trainSVC = None
            nextIndex = []

            for iSub in range(m):
                #print("Sub %d" %iSub)
                #print("Sub length: %d" %len(indexSub[iSub]))

                if (iSub % 2 == 0):
                    trainSVC = self.__get_svc(X[indexSub[iSub],], Y[indexSub[iSub],])
                    nextIndex = np.append(nextIndex, indexSub[iSub][trainSVC.support_])
                else:
                    yExpected = trainSVC.predict(X[indexSub[iSub],])

                    testErrorIndex = np.unique(np.nonzero((yExpected - Y[indexSub[iSub],]) != 0)[0])
                    testSuccessIndex = np.unique(np.nonzero((yExpected - Y[indexSub[iSub],]) == 0)[0])

                    delta = len(trainSVC.support_) / trainSize
                    #print(delta)

                    errorRatio = len(testErrorIndex) / (len(testErrorIndex) + len(testSuccessIndex))
                    correctRatio = 1 - errorRatio

                    nextIndex = np.append(nextIndex, indexSub[iSub][testErrorIndex[0 : testSize*errorRatio*delta]])
                    nextIndex = np.append(nextIndex, indexSub[iSub][testSuccessIndex[0 : testSize*correctRatio*delta]])

            nextIndex = [int(v) for v in nextIndex]

            X = X[nextIndex,]
            Y = Y[nextIndex,]
            n.append(X.shape[0])

            print("Number of SVs: %d / %d" % (n[i], n[i-1]), flush=True)
            print("Execute time (in second): %s" % (time.time() - startTime), flush=True)

            if  g*n[i]*i*beta*N/c >= (n[i-1]-n[i])**2:
                break

        svc = SVC(**self.svm_parameters)
        self.model = svc.fit(X,Y)
        return self.model

    def trainFileWithRatio(self, ratio, beta=0.01, g=1, debug=False):
        xTrain, yTrain = datasets.load_svmlight_file(svmlight_file_address)
        return self.trainWithRatio(ratio, xTrain, yTrain, beta, g, debug)

    def train_by_file(self, svmlight_file_address:str, beta=0.01, g=1, debug=False):
        xTrain, yTrain = datasets.load_svmlight_file(svmlight_file_address)
        return self.train_small(xTrain, yTrain, beta, g, debug)

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
