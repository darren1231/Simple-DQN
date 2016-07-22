function [ train_x,train_y ] = generate_sample(number)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
    
    number=10000;
    for i=1:number
    rand_x = floor(mod(rand*10,6))+1;
    rand_y = floor(mod(rand*10,6))+1;
    
    matrix = produce_state_picture(rand_x,rand_y);
    
    train_x(i,:)=matrix;
    index=6*(rand_x-1)+rand_y;
    zero_y= zeros(1,36);
    zero_y(index)=1;
    train_y(i,:)=zero_y;   
    
    end
    save('test_x.mat','train_x');
    save('test_y.mat','train_y');
end

