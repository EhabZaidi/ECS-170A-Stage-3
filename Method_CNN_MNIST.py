'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
from local_code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt


class Method_CNN_MNIST(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 10
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3
    batch_size = 128

    # it defines the the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        
        self.conv_layer_1 = nn.Conv2d(
            in_channels =1,
            out_channels = 16,
            kernel_size = 3,
            padding = 1
        )
        
        self.relu_1 = nn.ReLU()
        self.pool_1 = nn.MaxPool2d(kernel_size = 2, stride=2)
        
        self.conv_layer_2 = nn.Conv2d(
            in_channels = 16,
            out_channels = 32,
            kernel_size = 3,
            padding = 1
        )
        self.relu_2 = nn.ReLU()
        self.pool_2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.fc_layer_1 = nn.Linear(32*7*7, 128)
        self.relu_3 = nn.ReLU()
        
        self.fc_layer_2 = nn.Linear(128,10)

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, x):
        '''Forward propagation'''
        # hidden layer embeddings
        h = self.conv_layer_1(x)
        h = self.relu_1(h)
        h = self.pool_1(h)
        
        h = self.conv_layer_2(h)
        h = self.relu_2(h)
        h = self.pool_2(h)
        
        h = h.view(h.size(0), -1)
        
        h = self.fc_layer_1(h)
        h = self.relu_3(h)
        
        y_pred = self.fc_layer_2(h)
        return y_pred

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def train(self, X, y):
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.CrossEntropyLoss()
        # for training accuracy investigation purpose
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')
        loss_list = []
        accuracy_list = []
        
        X = torch.FloatTensor(np.array(X))
        y = torch.LongTensor(np.array(y))
        
        X = X.unsqueeze(1)
        
        X = X/255.0
        
        train_dataset = TensorDataset(X, y)
        train_loader = DataLoader(train_dataset, batch_size = self.batch_size, shuffle = True)

        # it will be an iterative gradient updating process
        # we don't do mini-batch, we use the whole input as one batch
        # you can try to split X and y into smaller-sized batches by yourself
        for epoch in range(self.max_epoch): # you can do an early stop if self.max_epoch is too much...
            total_loss = 0.0
            correct = 0
            total = 0
            
            for batch_X, batch_y in train_loader:
                y_pred = self.forward(batch_X)
                
                train_loss = loss_function(y_pred, batch_y)
                
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                
                total_loss += train_loss.item()
                
                pred_labels = y_pred.max(1)[1]
                correct += (pred_labels == batch_y).sum().item()
                total += batch_y.size(0)
            
            epoch_loss = total_loss / len(train_loader)
            epoch_accuracy = correct/total
            
            loss_list.append(epoch_loss)
            accuracy_list.append(epoch_accuracy)
            
            print('Epoch: ', epoch)
            print('Accuracy: ', epoch_accuracy)
            print('Loss: ', epoch_loss)
        
        plt.figure()
        plt.plot(loss_list)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('MNIST CNN Training Loss Convergence Curve')
        plt.savefig('mnist_cnn_training_loss_curve.png')
        plt.close()
        
        plt.figure()
        plt.plot(accuracy_list)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('MNIST CNN Training Accuracy Convergence Curve')
        plt.savefig('mnist_cnn_training_accuracy_curve.png')
        plt.close()
        
    def test(self, X):
        X = torch.FloatTensor(np.array(X))
        
        X = X.unsqueeze(1)
        
        X = X/255.0
        
        with torch.no_grad():
            y_pred = self.forward(X)
        
        return y_pred.max(1)[1]
    
    def run(self):
        print('method running...')
        print('--start training MNIST CNN...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing MNIST CNN...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
            