'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.setting import setting
import numpy as np

class Setting_Train_Test(setting):
    fold = 3
    
    def prepare(self, dataset_train, dataset_test, method, result, evaluate):
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.method = method
        self.result = result
        self.evaluate = evaluate
        
    
    def load_run_save_evaluate(self):
        
        # load dataset
        data = self.dataset_train.load()
        
        X_train = data['X_train']
        y_train = data['y_train']
        
        X_test = data['X_test']
        y_test = data['y_test']
        
        self.method.data = {
            'train' : {'X' : X_train, 'y': y_train},
            'test' : {'X' : X_test, 'y' : y_test}
        }
        
        learned_result = self.method.run()
            
        # save raw ResultModule
        self.result.data = learned_result
        self.result.fold_count = 1
        self.result.save()
            
        self.evaluate.data = learned_result
        
        return self.evaluate.evaluate(), None

        