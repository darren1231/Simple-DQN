%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ---------------------------------------------------------------------------------
% Project <Deep Q learning with autoencoder>
% Date    : 2016/01/27
% Author  : Kun da Lin
% Comments: Language: Matlab. 
% Source: matlab 
% ---------------------------------------------------------------------------------
% map_matrix = [start,1,1,1,1,1;
%               1,0,0,0,0,1;
%               1,0,0,0,0,1;
%               1,0,0,0,0,1;
%               1,0,0,0,0,1;
%               1,1,1,1,1,goal];
%Q(state,x1)=  oldQ + alpha * (R(state,x1)+ (gamma * MaxQ(x1)) - oldQ);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;
%% load pretrain atuto-encoder data
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
%%%%%%%%%%%%%%%%%%%%

[experience,train_experience,success,fail]=collect_experience();

train_x=experience(1:19,:);

net_low = newff(minmax(train_x),[19,10,1],{'logsig','logsig','logsig'},'trainrp');
net_low = init(net_low);
% [ max_matrix ] = compute_max(nn_low,train_experience,success,fail);

% for i=1:20
% [ max_matrix ] = compute_max(net_low,train_experience,success,fail);
% [ net_low ] = iteration( train_x,max_matrix);
% disp(i);
% end

% matrix=produce_state_picture( 1,1 );
% output= nn_compute_output( nn,matrix ); 
% low_feature=repmat(output',1,4);
% action_feature=[1 0 0 0;0 1 0 0 ;0 0 1 0;0 0 0 1];
% conbine_two=[low_feature;action_feature];
% nn_out=sim(net_low,conbine_two);
% [max_q, max_index] = max(nn_out);

iteration_number=0;
epochs=0;
frame=0;   
 while(epochs<20)
    position_x=1;   %initial position
    position_y=1;
    step=0;
    epochs=epochs+1;
      [ max_matrix ] = compute_max(net_low,train_experience,success,fail);
      [ net_low ]    = iteration( train_x,max_matrix); 
    Episode_step=0;  
while ~(position_x==6 && position_y==6)
    
    Episode_step = Episode_step+1
    
    if(step>10)
        
      iteration_number=iteration_number+1;
      disp(['run_iteration:  ' num2str(iteration_number)]);
      
      [ max_matrix ] = compute_max(net_low,train_experience,success,fail);
      [ net_low ]    = iteration( train_x,max_matrix);  
      step=0;
      position_x=1;   %initial position
      position_y=1;      
    end
    step=step+1;
    
       %% Show picture
    frame=frame+1;
    matrix=produce_state_picture( position_x,position_y );
    I=reshape(matrix,30,30);
%     imshow(I);
%     imshow(I,[]); 
    imshow(I,'InitialMagnification','fit');
    Map(frame)=getframe;
       %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    output= nn_compute_output( nn,matrix );
    
    low_feature=repmat(output',1,4);
    action_feature=[1 0 0 0;0 1 0 0 ;0 0 1 0;0 0 0 1];
    conbine_two=[low_feature;action_feature];
    nn_out=sim(net_low,conbine_two);
    [max_q, max_index] = max(nn_out);
    
    pre_position_x=position_x;
    pre_position_y=position_y;
    
    switch max_index                           % step4:implement the policy
     
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
     end
     if(position_x==6 && position_y==6)
             %% Show picture
        matrix=produce_state_picture( position_x,position_y );
        I=reshape(matrix,30,30);
        imshow(I);
        imshow(I,[]); 
        imshow(I,'InitialMagnification','fit');
        Map(frame)=getframe;
     end
      disp(['x:' num2str(position_x) '   y:' num2str(position_y)]);
      disp(['step: ' num2str(step)]);   
end
      disp('')
      disp(['Episode:' num2str(epochs) '  Step:' num2str(Episode_step) ]);
end
%movie2avi(Map,'slow','compression', 'None','FPS',30);