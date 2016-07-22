function [ net ] = iteration( input,output )
%UNTITLED10 Summary of this function goes here
%   Detailed explanation goes here



% % Train the FFNN
% opts.numepochs =   100;
% opts.batchsize = 100;
% nn_low = nntrain(nn_low, train_x', max_matrix, opts);

net = newff(minmax(input),[19,10,1],{'logsig','logsig','logsig'},'trainrp');
net.trainParam.showWindow = false;
net.trainParam.showCommandLine = false;
net = train(net,input,output);
end

