'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.dataset import dataset
import numpy as np
import pickle


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
    
    def load(self):
        print('loading image data...')
        
        file_path = self.dataset_source_folder_path + self.dataset_source_file_name
        
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            
        X_train = []
        y_train = []
        X_test = []
        y_test = []
        
        for instance in data['train']:
            image = np.array(instance['image'])
            label = instance['label']
            
            X_train.append(image)
            y_train.append(label)
            
        for instance in data['test']:
            image = np.array(instance['image'])
            label = instance['label']
            
            X_test.append(image)
            y_test.append(label)
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        
        return{
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test
        }