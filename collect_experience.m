function [experience,train_experience,success,fail]=collect_experience()

load('autoencoder_w_900 484 225 121 113 57 29 15.mat');
  
nn = nnsetup([900 484 225 121 113 57 29 15]);
nn.activation_function              = 'sigm';
nn.learningRate                     = 1;

nn.W{1}=w1 ;
nn.W{2}=w2 ;
nn.W{3}=w3 ;
nn.W{4}=w4 ;
nn.W{5}=w5 ;
nn.W{6}=w6 ;
nn.W{7}=w7 ;

% x=zeros(2,900);
epochs=0;
i=0;
j=0;
k=0;
m=0;
while epochs<10
epochs=epochs+1;    
position_x=1;       %initial position                                    
position_y=1;
first_time=0;
while ~(position_x==6 && position_y==6)     %   set goal station
i=i+1; 
j=j+1;
pre_position_x=position_x;
pre_position_y=position_y;

matrix=produce_state_picture( position_x,position_y );
output= nn_compute_output( nn,matrix );     
rand_action = floor(mod(rand*10,4))+1;
train_experience(:,j)=[output'];

if(first_time==0)
    j=j-1;
    first_time=1;
end
    switch rand_action                           % step4:implement the policy
     
    case 1
        position_y = pre_position_y-1;  %up
    case 2
        position_y = pre_position_y+1;  %down
    case 3
        position_x = pre_position_x-1;  %left
    case 4
        position_x = pre_position_x+1;  %right
    end
    
     if(position_x==0 || position_x==7 || position_y==0 || position_y==7)
        position_x = pre_position_x;
        position_y = pre_position_y;
        reward=-1;
        m=m+1;
        fail(m)=i;
        
     elseif(position_x==6 && position_y==6)
        reward=1;
        j=j+1;
        train_experience(:,j)=ones(15,1);
        
        k=k+1;
        success(k)=i;        
     else
        reward=0;
     end
     
     if(rand_action==1)
         action_table=[1;0;0;0];
     elseif(rand_action==2)
         action_table=[0;1;0;0];
     elseif(rand_action==3)
         action_table=[0;0;1;0];
     else
         action_table=[0;0;0;1];
     end
         
     
     experience(:,i)=[output';action_table;position_x;position_y;reward];
     %disp(['x:' num2str(position_x) '   y:' num2str(position_y)]);
end
     disp('Collect experience');
     disp(epochs);
end
