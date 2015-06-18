# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 05:49:08 2015

@author: viethoangcr
"""

import numpy as np
import collections
import scipy.sparse as sp
import tempfile
import os

#file_address = r'./../dataset/mnist_train_784_poly_8vr.dat'
file_address = r'./a1a.txt'

class SvmlightFileHandler(object):
    """
    This class is a handler of svmlight data file, which is use a large data
    file and does not load whole file to RAM when using tempfile instead
    """
    def __init__(self, file_address:str, temp_folder:str = None,
                 n_features:int = None, multilabel:bool = False,
                 zero_based='auto'):
        self.file_address = file_address
        if (temp_folder is not None) and (os.access(temp_folder, os.W_OK)):
            tempfile.tempdir = temp_folder
        
        self.temp_file = ""
        self.originalInputFile = True
        
        self.line_offset = []
        self.n_features = n_features
        self.multilabel = multilabel        
        self.zero_based = zero_based
        
        self.size = 0
        self.__analyze()
        
    # indexing offset of each line        
    def __analyze(self):
        n_features = 0
        lowest_index = 2
        with open(self.file_address) as file:
            offset = 0
            for line in file:
                position = offset              
                offset += len(line)
                # skip comments
                line = line[:line.find('#')]
                line_parts = line.split()
                
                if len(line_parts) == 0:
                    continue
                
                if self.n_features is None:
                    for i in range(1, len(line_parts)):
                        f, v = line_parts[i].split(':',1)
                        n_features = max(n_features, int(f))
                        lowest_index = min(lowest_index, int(f))
                        
                self.line_offset.append(position)

            file.close()
            
            # check for index based in the file
            if self.zero_based == 'auto':
                # if file is zero based
                if lowest_index == 0:
                    n_features += 1
                    self.zero_based = True
                elif lowest_index > 0:
                    self.zero_based = False
                else:
                    raise ValueError(
                        "Invalid index %d in SVMlight/LibSVM data file." 
                        % lowest_index)
            self.size = len(self.line_offset)
        
        if self.n_features is None:
            self.n_features = n_features
            
    def __getitem__(self, indexes):
        data = []
        indptr = [0]
        indices = []
        labels = []
        
        # lamba function to process None value
        ifnone = lambda a, b: b if a is None else a
        # check if indexes is a slice object or not
        if isinstance(indexes, slice):
            indexes = range(ifnone(indexes.start, 0), ifnone(indexes.stop,
                            self.size), ifnone(indexes.step, 1))
            
        if not(isinstance(indexes, collections.Iterable)):
            indexes = [indexes]
            
        if len(indexes) > 0:
            with open(self.file_address) as file:
                for k in indexes:
                    if (k < 0) or (k > self.size):
                        continue
                    else:
                        file.seek(self.line_offset[k])
                        
                        # get line
                        line = file.readline()
                        pos = line.find('#')
                        if pos >= 0:
                            line = line[:pos]
                        line_parts = line.split()
                        target, features = line_parts[0], line_parts[1:]                        
                        
                        # process for multi label
                        if self.multilabel:
                            target = [float(y) for y in target.split(',')]
                            target.sort()
                            labels.append(tuple(target))
                        else:
                            labels.append(float(target))
                        
                        #TODO: implement qid
                        
                        for feature in features:
                            idx_s, v = feature.split(':',1)          
                            indices.append(int(idx_s))
                            data.append(float(v))
                        indptr.append(len(data))
                        
                file.close()

        indices = np.array(indices)
        # convert to zero based (python index) if not
        if not self.zero_based:
            indices -= 1
        shape = (len(indptr) -1, self.n_features)

        result = sp.csr_matrix((np.array(data), indices, np.array(indptr)), shape)
        result.sort_indices()
        return (result, labels)
        
    def filter(self, indexes):
        # lamba function to process None value
        ifnone = lambda a, b: b if a is None else a
        # check if indexes is a slice object or not
        if isinstance(indexes, slice):
            indexes = range(ifnone(indexes.start, 0), ifnone(indexes.stop,
                            self.size), ifnone(indexes.step, 1))
            
        if not(isinstance(indexes, collections.Iterable)):
            indexes = [indexes]
            
        if len(indexes) > 0:
            # new line offset
            line_offset = []
            offset = 0
            # create a temp file
            tf = tempfile.NamedTemporaryFile(mode="w", delete=False)
            self.temp_file = tf.name
            with open(self.file_address) as file:
                for k in indexes:
                    # first check condition of k
                    if (k < 0) or (k > self.size):
                        continue
                    else:
                        # jump to that line
                        file.seek(self.line_offset[k])
                        position = offset
                        # read line, remove comment if exists then write to temp
                        line = file.readline()
                        pos = line.find('#')
                        if pos >= 0:
                            line = line[:line.find('#')] + '\n'
                        tf.write(line)
                        offset += len(line)
                        line_offset.append(position)
                        
                self.line_offset = line_offset
                self.size = len(self.line_offset)
                if not(self.originalInputFile):
                    os.remove(self.file_address)
                self.file_address = self.temp_file
                self.originalInputFile = False
            tf.close()
                        
            
    
    def __del__(self):
        if os.path.exists(self.temp_file):
            os.remove(self.temp_file)
        if not(self.originalInputFile) and os.path.exists(self.file_address):
            os.remove(self.file_address)
            
    def print_info(self):
        print('Number of features: %d ' % self.n_features)
        print('Number of entries:  %d ' % len(self.line_offset))
        