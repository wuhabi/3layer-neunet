% Codes by Haibing Wu, Dept.of E.E., Fudan University, imwuhaibing@gmail.com

function [model] = bptrain(layers, train_x, train_y,epochs)
% This function trains a 3 layers neural networks with backpropogation 
% algorithm under mini-batch mode with bias.
% Inputs:
%   train_x - input data matrix, is of size (#instances, #features)
%   train_y - targets for input data, is of size (#instances, #labels)
% Output:
%   model - A struct containing the learned weights

    %%% Set Hyper Parameters %%%
    numint = layers(1); % number of input units
    numhid = layers(2); % number of hidden units
    numout = layers(3); % number of softmax output units
    learning_rate = 0.1;
    batchsize = 100; 
    numbatches = 600;
    
    %%% Initialization %%%
    int2hid_weights = 0.1*randn(numint, numhid);
    hid2out_weights = 0.1*randn(numhid, numout);
    hidbias = zeros(numhid, 1);
    outbias = zeros(numout, 1);
    
    %%% Make batches %%%
    inputbatches = reshape(train_x', numint, batchsize, numbatches);
    targetbatches = reshape(train_y', numout, batchsize, numbatches);
    
    % Loop over epochs %
    for epoch = 1:epochs
       fprintf(1, 'epoch %d\n', epoch);
       aver_ce = 0;
       tic;
       % Loop over batches %
       for batch = 1:numbatches
            batchinput = inputbatches(:,:,batch)'; % size (batchsize X numdim) 
            batchtarget = targetbatches(:,:,batch)';
            
            %%% Step 1: Forward Propagation %%%
            % Compute state of each layer in the networks given the input
            % batch and all weights and biases
            
            % Inputs to hidden units, with size (#batch cases, #hiden units)
            inputs2hid = batchinput*int2hid_weights + repmat(hidbias, 1, batchsize)'; 
            % States of hidden units                                         
            hidstates = 1./(1+exp(-inputs2hid));
            
            % Inputs to softmax output units, with size (#batch cases, #output units)
            inputs2out = hidstates*hid2out_weights + repmat(outbias, 1, batchsize)';    
            % Subtract maxima. 
            % Adding or subtracting the same constant from each input to a
            % softmax unit does not affect the outputs. Here we are subtracting maximum to
            % make all inputs <= 0. This prevents overflows when computing their exponents.
            inputs2out = inputs2out - repmat(max(inputs2out,[],2), 1, numout);
            
            % states of softmax units, with size (#batch cases, #softmax output units)
            outstates = exp(inputs2out);
            % Normalize to get probability distribution
            outstates = outstates./repmat(sum(outstates,2),1,numout);
            
            %%% Measure Cross Entropy Loss %%% 
            tiny = exp(-30);
            % current batch entropy loss
            curbatch_ce = -sum(sum(batchtarget.*log(outstates+tiny),2))/numbatches;
            aver_ce = aver_ce + (curbatch_ce-aver_ce)/batch;
            % fprintf(1,'\rBatch %d Train CE %.3f', batch, aver_ce);
            
            %%% Step 2: Back Propagation %%%
            % Backpropagation is employed to compute derivative of loss
            % function w.r.t weights in neural network.
            
            % 2(a): First compute deltas
            % partial derivative of cross entropy loss function w.r.t inputs to softmax units 
            inputs2out_deriv = outstates - batchtarget; 
            % partial derivative of cross entropy loss function w.r.t inputs to hidden units
            inputs2hid_deriv = inputs2out_deriv*hid2out_weights'.*hidstates.*(1-hidstates);
            
            % 2(b): Then compute gradients
            % partial derivative of cross entropy loss function w.r.t hid2out_weights
            hid2out_weights_grad = hidstates'*inputs2out_deriv; % size (#hidden layers, #output layers)
            % partial derivative of cross entropy loss function w.r.t outbias
            outbias_grad = sum(inputs2out_deriv, 1)';
            % partial derivative of cross entropy loss function w.r.t int2hid_weights
            int2hid_weights_grad = batchinput'*inputs2hid_deriv;
            % partial derivative of cross entropy loss function w.r.t % hidbias
            hidbias_grad = sum(inputs2hid_deriv, 1)';
            
            %%% Step3: Update Weights %%%
            % weights are updated based on gradient descent
            int2hid_weights_delta = int2hid_weights_grad/numbatches;
            int2hid_weights = int2hid_weights - learning_rate*int2hid_weights_delta;
            hidbias_delta = hidbias_grad/numbatches;
            hidbias = hidbias - learning_rate*hidbias_delta;
            
            hid2out_weights_delta = hid2out_weights_grad/numbatches;
            hid2out_weights = hid2out_weights - learning_rate*hid2out_weights_delta;
            outbias_delta = outbias_grad/numbatches;
            outbias = outbias - learning_rate*outbias_delta;
        end
        toc;
    end
    model.int2hid_weights = int2hid_weights;
    model.hidbias = hidbias;
    model.hid2out_weights = hid2out_weights;
    model.outbias = outbias;
end

