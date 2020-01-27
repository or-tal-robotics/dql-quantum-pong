clc; close all; clear;
% addpath('C:\Users\bar\Google Drive\bar_n_amit\matlab\QetLab');
Stat = readtable('stat2.csv','Delimiter','\t');
%Stats_weights_left = table2array(readtable('weights_stats_left.csv'));
%Stats_weights_right = table2array(readtable('weights_stats_right.csv'));



Pauli{1} = [0 1; 1 0];
Pauli{2} = [0 -1i; 1i 0];
Pauli{3} = [1 0; 0 -1];
Pauli{4} = [1,0;0,1];
ONE=[1,0;0,1];
Action_right=7;
Action_left=5;

%AAA=table2array(weights_Stat(1,1));

for i=1:20%prod(size(Stat(:,1)))
    
    theta_Left1=table2array(Stat(i,2));
    theta_Left2=table2array(Stat(i,3));
    
    alpha_ent=table2array(Stat(i,4));
    theta_right1=table2array(Stat(i,5));
    theta_right2=table2array(Stat(i,6));
        
    psi_left=str2num(cell2mat(table2array(Stat(i,7))));
    psi_right=str2num(cell2mat(table2array(Stat(i,8))));
    left_crystal=table2array(Stat(i,9))/max(table2array(Stat(:,9)));
    right_crystal=table2array(Stat(i,10))/max(table2array(Stat(:,10)));
    win=table2array(Stat(i,11));
    ball_left=table2array(Stat(i,12));
    ball_right=table2array(Stat(i,13));
    left_crystal_p=table2array(Stat(i,14));
    right_crystal_p=table2array(Stat(i,15));
    
    if sin(ball_left)>0
        Left_devices=sin(theta_Left1)*Pauli{3}+cos(theta_Left1)*Pauli{1};
    else
        Left_devices=sin(theta_Left2)*Pauli{3}+cos(theta_Left2)*Pauli{1};
        
    end
    if sin(ball_right)<0
        right_devices=sin(theta_right1)*Pauli{3}+cos(theta_right1)*Pauli{1};

    else
        right_devices=sin(theta_right2)*Pauli{3}+cos(theta_right2)*Pauli{1};

    end    
    
    psi_befor =[cos(alpha_ent),0,0,sin(alpha_ent)]';
    
    psi_left*psi_befor;
    if round((psi_left*psi_befor)-1)
        i
        psi_left*psi_befor
    end
    %  psi =simplify(psi /norm(psi));
    Density_AB=psi_befor*psi_befor';
    
    Corr_AB=(trace(Density_AB*kron(Left_devices,right_devices)));
    Corr_AA=(trace(Density_AB*kron(Left_devices,ONE)));
    befor_crystal_p=(1+Corr_AA)/2;
    befor_crystal_p/left_crystal_p;
    %[u,v]=  eig(kron(Left_devices,ONE));
    [U,V]=  eig(Left_devices);
    U1=kron(U(:,1),[1,0]')';
    U2=kron(U(:,1),[0,1]')';
    U3=kron(U(:,2),[1,0]')';
    U4=kron(U(:,2),[0,1]')';
    
    
    if round(left_crystal)==0
        psi_after=(U1'*U1+U2'*U2)*psi_befor;
        psi_after=psi_after/norm(psi_after);
        
    else
        
        psi_after=(U3'*U3+U4'*U4)*psi_befor;
        psi_after=psi_after/norm(psi_after);
        
    end
    
    
    if round((psi_right*psi_after)-1,3)
        i
        psi_right*psi_after
        
    end
    
    Density_AB=psi_after*psi_after';
    
    Corr_BB=(trace(Density_AB*kron(ONE,right_devices)));
    after_crystal_p=(1+Corr_BB)/2;
    
    if round((after_crystal_p/right_crystal_p)-1,3)
        i
        after_crystal_p/right_crystal_p
    end
    
    
    
end


left_crystal_vec=2*table2array(Stat(:,9))/max(table2array(Stat(:,9)))-1;
right_crystal_vec=2*table2array(Stat(:,10))/max(table2array(Stat(:,10)))-1;
win_vec=(table2array(Stat(:,11))+1)/2;
ball_left_vec=sin(table2array(Stat(:,12)));
ball_left_vec=ball_left_vec/max(ball_left_vec);
ball_right_vec=sin(table2array(Stat(:,13)));
ball_right_vec=ball_right_vec/max(ball_right_vec);

sam=200;
MM=20;
corr_crystal=left_crystal_vec.*right_crystal_vec;
corr_num=round((4*atan2(ball_left_vec,ball_right_vec)/pi+3)/2+1);

table=[left_crystal_vec,right_crystal_vec,corr_crystal,corr_num,win_vec];

mean(table( find(table(:,4)==2 &table(:,3)==1),5));%
%histogram(corr_num)
num_1=find(corr_num==1);
num_2=find(corr_num==2);
num_3=find(corr_num==3);
num_4=find(corr_num==4);

%num_1(find(num_1<=1000 & num_1>=500))
for  j=sam:sam:length(win_vec)
    jj=j/sam;
    mean_win_vec(jj)=mean(win_vec(1:j));
    
    mean_win_vec_sam(jj)=mean(win_vec(j-sam+MM:j));
    
    mean_corr_crystal_1(jj)=   mean(corr_crystal(num_1(find(num_1<=j & num_1>=(j-sam+MM)))));
    mean_corr_crystal_2(jj)=   mean(corr_crystal(num_2(find(num_2<=j & num_2>=(j-sam+MM)))));
    mean_corr_crystal_3(jj)=   mean(corr_crystal(num_3(find(num_3<=j & num_3>=(j-sam+MM)))));
    mean_corr_crystal_4(jj)=   mean(corr_crystal(num_4(find(num_4<=j & num_4>=(j-sam+MM)))));
    
    bell(jj)=-mean_corr_crystal_1(j/sam)+mean_corr_crystal_2(j/sam)...
        +mean_corr_crystal_3(j/sam)+mean_corr_crystal_4(j/sam);
    
    bell2(jj)=mean_corr_crystal_1(j/sam)-mean_corr_crystal_2(j/sam)...
        +mean_corr_crystal_3(j/sam)+mean_corr_crystal_4(j/sam);
    
    bell3(jj)=mean_corr_crystal_1(j/sam)+mean_corr_crystal_2(j/sam)...
        -mean_corr_crystal_3(j/sam)+mean_corr_crystal_4(j/sam);
    
    bell4(jj)=mean_corr_crystal_1(j/sam)+mean_corr_crystal_2(j/sam)...
        +mean_corr_crystal_3(j/sam)-mean_corr_crystal_4(j/sam);
end
t=1:1:length(win_vec)/sam;

% windowSize = 10;
% b = (1/windowSize)*ones(1,windowSize);
% a = 1;
%mean_win_vec_sam = filter(b,a,mean_win_vec_sam);
%bell4=filter(b,a,bell4);

figure(1)
plot(t,mean_win_vec,'b',t,mean_win_vec_sam,'r')
figure(2)

plot(t,mean_corr_crystal_1,'b',t,mean_corr_crystal_2,'g',t,mean_corr_crystal_3,'k',t,mean_corr_crystal_4,'r');

figure(3)
plot(t,bell4,'r')%t,bell,'b',t,bell2,'g',t,bell3,'k',

% A = randn(10,2);
% B = randn(10,1);
% R = corrcoef([A,B])
%R{1}=corrcoef([Stats_weights_left(1:400,2:2561),Stats_weights_right(1:400,2:3584)]);
%Corr_brain=cell(length(win_vec)/sam,1);
%Stats_weights_left=csvread('weights_stats_left.csv',100,0,[100,0,200,521])

for  j=1:1:length(win_vec)/sam
    
Stats_weights_left=csvread('weights_stats_left.csv',j*2*sam-2*sam+1,0,[j*2*sam-2*sam+1, 0 ,j*2*sam ,512*Action_left]);
Stats_weights_right=csvread('weights_stats_right.csv',j*2*sam-2*sam+1,0,[j*2*sam-2*sam+1 ,0, j*2*sam ,512*Action_right]);


   Corr_brain = cov([Stats_weights_left,Stats_weights_right]);
   %Corr_brain1=Corr_brain((512*Action_left+1):512*(Action_left+Action_right),...
       %(512*Action_left+1):512*(Action_left+Action_right));
  %[row, col] = find(isnan(Corr_brain1))
  % [~,S_left,~] = svd(Corr_brain(1:512*Action_left,1:512*Action_left));
 %  [~,S_right,~] = svd(Corr_brain1);
   [~,S_both,~] = svd(Corr_brain(1:512*Action_left,512*Action_left+1:end));
%trace(S_left)
%trace(S_right)
SS_both(j)=sum(diag(S_both));

tt=1:1:j;

figure(4)

plot(tt,SS_both,'b')
ylim auto
hold off
drawnow 
end



