function []=predict(id)
test_X=dlmread(strcat("temp/",id,"-predict_X.csv"), "\t", 1, 1);
load(strcat("temp/",id,"-model.mat"))

if feature_selection == "True"
    temp=test_X;
    test_x=temp(:,logical(optimal_parameter));
else
    test_x=test_X;
end

predict_matrix=mlknn_test(train_x,train_y,test_x,neigbor_num,Prior,PriorN,Cond,CondN);
dlmwrite(strcat("temp/",id,"-predict_y.csv"), predict_matrix, "\t")
end

function Outputs=mlknn_test(train_data,train_target,test_data,Num,Prior,PriorN,Cond,CondN)
train_target(train_target==0)=-1;
train_target=train_target';

%MLKNN_test tests a multi-label k-nearest neighbor classifier.
%
%    Syntax
%
%       [HammingLoss,RankingLoss,OneError,Coverage,Average_Precision,Outputs,Pre_Labels]=MLKNN_test(train_data,train_target,test_data,test_target,Num,Prior,PriorN,Cond,CondN)
%
%    Description
%
%       KNNML_test takes,
%           train_data       - An M1xN array, the ith instance of training instance is stored in train_data(i,:)
%           train_target     - A QxM1 array, if the ith training instance belongs to the jth class, then train_target(j,i) equals +1, otherwise train_target(j,i) equals -1
%           test_data        - An M2xN array, the ith instance of testing instance is stored in test_data(i,:)
%           test_target      - A QxM2 array, if the ith testing instance belongs to the jth class, test_target(j,i) equals +1, otherwise test_target(j,i) equals -1
%           Num              - Number of neighbors used in the k-nearest neighbor algorithm
%           Prior            - A Qx1 array, for the ith class Ci, the prior probability of P(Ci) is stored in Prior(i,1)
%           PriorN           - A Qx1 array, for the ith class Ci, the prior probability of P(~Ci) is stored in PriorN(i,1)
%           Cond             - A Qx(Num+1) array, for the ith class Ci, the probability of P(k|Ci) (0<=k<=Num) i.e. k nearest neighbors of an instance in Ci will belong to Ci , is stored in Cond(i,k+1)
%           CondN            - A Qx(Num+1) array, for the ith class Ci, the probability of P(k|~Ci) (0<=k<=Num) i.e. k nearest neighbors of an instance not in Ci will belong to Ci, is stored in CondN(i,k+1)
%      and returns,
%           HammingLoss      - The hamming loss on testing data
%           RankingLoss      - The ranking loss on testing data
%           OneError         - The one-error on testing data as
%           Coverage         - The coverage on testing data as
%           Average_Precision- The average precision on testing data
%           Outputs          - A QxM2 array, the probability of the ith testing instance belonging to the jCth class is stored in Outputs(j,i)
%           Pre_Labels       - A QxM2 array, if the ith testing instance belongs to the jth class, then Pre_Labels(j,i) is +1, otherwise Pre_Labels(j,i) is -1

num_class=size(train_target,1);
num_training=size(train_target,2);
num_testing=size(test_data,1);
%Computing distances between training instances and testing instances
dist_matrix=pdist2(test_data,train_data);
Neighbors=zeros(num_training,Num); %Neighbors{i,1} stores the Num neighbors of the ith training instance
[temp,index]=sort(dist_matrix,2);
Neighbors=index(:,1:Num);
Outputs=zeros(num_class,num_testing);
for i=1:num_testing
    temp=zeros(1,num_class); %The number of the Num nearest neighbors of the ith instance which belong to the jth instance is stored in temp(1,j)
    neighbor_labels=[];
    for j=1:Num
        neighbor_labels=[neighbor_labels,train_target(:,Neighbors(i,j))];
    end
    
    temp=sum(neighbor_labels==ones(num_class,Num),2);
    temp=temp';
    
    for j=1:num_class
        Prob_in=Prior(j)*Cond(j,temp(1,j)+1);
        Prob_out=PriorN(j)*CondN(j,temp(1,j)+1);
        if(Prob_in+Prob_out==0)
            Outputs(j,i)=Prior(j);
        else
            Outputs(j,i)=Prob_in/(Prob_in+Prob_out);
        end
    end
end
Outputs=Outputs';
end