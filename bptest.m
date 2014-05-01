% Codes by Haibing Wu, Dept.of E.E., Fudan University, imwuhaibing@gmail.com

function [preds, acc] = bptest(model, test_x, test_y)
% Input
%   model: A structure conaining weights optimized by backpropagation
%   test_x: input data, each row correponds to features of a test case
%   test_y: targets for input data
% Output
%   preds: predicted labels
%   acc: accuracy, i.e. (#corrected predicted cases)/(#total test cases)
    
    numcase = size(test_x, 1);
    int2hid_weights = model.int2hid_weights;
    hid2out_weights = model.hid2out_weights;
    hidbias = model.hidbias;
    outbias = model.outbias;
    
    numcorrected = 0;
    preds = zeros(numcase, 1);
    
    for m = 1:numcase
        input = test_x(m,:);
        target = test_y(m,:);
        
        % Perform forward propagation
        inputs2hid = input*int2hid_weights + hidbias';
        hidstates = 1./(1+exp(-inputs2hid));
        
        intputs2out = hidstates*hid2out_weights + outbias';
        outstate = exp(intputs2out);
        outstate = outstate./sum(outstate);
        
        [C1, I1] = max(outstate);
        [C2, I2] = max(target);
        if I1 == I2
            numcorrected = numcorrected + 1;
        end
        preds(m, 1) = I1-1;
    end
    acc = numcorrected/numcase;
end

