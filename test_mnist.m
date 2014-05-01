% Codes by Haibing Wu, Dept.of E.E., Fudan University, imwuhaibing@gmail.com

% This program trains a 784-3000-10 feed forward neural network on mnist 
% hand written digit data set using backpropagation algorithm. 
% Run with 1 epoch, will get around 87% accuracy,with 100 epoch will get 
% around 97% accuracy.

clear all; clc;

% Load data set
load mnist_uint8;
train_x = double(train_x)/255; train_y = double(train_y);
test_x = double(test_x)/255; test_y = double(test_y);

% Set architechture
layers = [784, 3000, 10];

% Train a 3 layers feed forward neural network with backpropagation
model = bptrain(layers, train_x, train_y,1);

% Testing
[predlabels, acc] = bptest(model,test_x,test_y);

fprintf(1,'Accuracy:%.4f\n', acc);