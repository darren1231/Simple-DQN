%function test_example_SAE
load mnist_uint8;

train_x = double(train_x)/255;
test_x  = double(test_x)/255;
train_y = double(train_y);
test_y  = double(test_y);

% normalize
[train_x, mu, sigma] = zscore(train_x);
test_x = normalize(test_x, mu, sigma);

%%  ex1 train a 100 hidden unit SDAE and use it to initialize a FFNN
%   Setup and train a stacked denoising autoencoder (SDAE)
rand('state',0)
sae = saesetup([784 500 300]);
sae.ae{1}.activation_function       = 'sigm';
sae.ae{1}.learningRate              = 1;
sae.ae{1}.inputZeroMaskedFraction   = 0.5;
sae.ae{2}.activation_function       = 'sigm';
sae.ae{2}.learningRate              = 1;
sae.ae{2}.inputZeroMaskedFraction   = 0.5;
% sae.ae{3}.activation_function       = 'sigm';
% sae.ae{3}.learningRate              = 1;
% sae.ae{3}.inputZeroMaskedFraction   = 0.5;
opts.numepochs =   1;
opts.batchsize = 100;
sae = saetrain(sae, train_x, opts);
%visualize(sae.ae{1}.W{1}(:,2:end)')

% Use the SDAE to initialize a FFNN
vx   = train_x(1:10000,:);
tx = train_x(10001:end,:);
vy   = train_y(1:10000,:);
ty = train_y(10001:end,:);

nn = nnsetup([784 500 300 10]);
%nn.activation_function              = 'sigm';
nn.learningRate                     = 1;
nn.output               = 'softmax'; 
nn.W{1} = sae.ae{1}.W{1};
nn.W{2}=sae.ae{2}.W{1};
%nn.W{3}=sae.ae{3}.W{1};
% Train the FFNN
opts.numepochs =  7;
opts.batchsize = 1000;
opts.plot      = 1;            %  enable plotting

nn = nntrain(nn, tx, ty, opts, vx, vy);       %  nntrain takes validation set as last two arguments (optionally)
[er, bad] = nntest(nn, test_x, test_y);
disp(er);
%assert(er < 0.16, 'Too big error');

% %% compare to normal one
% % normalize
% % [train_x, mu, sigma] = zscore(train_x);
% % test_x = normalize(test_x, mu, sigma);
% 
% rand('state',0)
% vx   = train_x(1:10000,:);
% tx = train_x(10001:end,:);
% vy   = train_y(1:10000,:);
% ty = train_y(10001:end,:);
% 
% nn = nnsetup([784 20 10]);
% %nn.activation_function              = 'sigm';
% nn.output               = 'softmax';                   %  use softmax output
% nn.learningRate                     = 1;
% opts.numepochs =   50;
% opts.batchsize = 1000;
% opts.plot      = 1;            %  enable plotting
% nn = nntrain(nn, tx, ty, opts, vx, vy);                %  nntrain takes validation set as last two arguments (optionally)
% [er1, bad1] = nntest(nn, test_x, test_y);
% disp(er1);

