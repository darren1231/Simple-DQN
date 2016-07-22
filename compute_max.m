function [ max_matrix ] = compute_max(net_low,train_experience,success,fail)
%UNTITLED8 Summary of this function goes here
%   Detailed explanation goes here
% [experience,train_experience,success]=collect_experience();
% 
% nn_low = nnsetup([19 10 1]);
% nn_low.activation_function              = 'sigm';
% nn_low.learningRate                     = 1;

action_up=repmat([1;0;0;0],1,size(train_experience,2));
action_down=repmat([0;1;0;0],1,size(train_experience,2));
action_left=repmat([0;0;1;0],1,size(train_experience,2));
action_right=repmat([0;0;0;1],1,size(train_experience,2));

train_experience_up=[train_experience;action_up];
train_experience_down=[train_experience;action_down];
train_experience_left=[train_experience;action_left];
train_experience_right=[train_experience;action_right];


output(1,:)=sim(net_low,train_experience_up);
output(2,:)=sim(net_low,train_experience_down);
output(3,:)=sim(net_low,train_experience_left);
output(4,:)=sim(net_low,train_experience_right);
% output(:,1) = nn_compute_output( nn_low,train_experience_up');
% output(:,2) = nn_compute_output( nn_low,train_experience_down');
% output(:,3) = nn_compute_output( nn_low,train_experience_left');
% output(:,4) = nn_compute_output( nn_low,train_experience_right');

max_matrix=max(output,[],1);
max_matrix=0.9*max_matrix;
for i=1:size(success,2)
    max_matrix(success(i))=1;
end
for i=1:size(fail,2)
    max_matrix(fail(i))=0;
end

end

