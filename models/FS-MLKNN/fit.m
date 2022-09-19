function []=fit(id,neigbor_num,feature_selection)
train_X=dlmread(strcat("temp/",id,"-fit_X.csv"), "\t", 1, 1);
train_y=dlmread(strcat("temp/",id,"-fit_y.csv"), "\t", 1, 1);
neigbor_num=str2double(neigbor_num);

if feature_selection == "True"
    [optimal_parameter,self_predict_score]=get_parameter_train(train_X,train_y);
    temp=train_X;
    train_x=temp(:,logical(optimal_parameter));
else
    train_x = train_X;
end

[Prior,PriorN,Cond,CondN]=mlknn_train(train_x,train_y,neigbor_num,1);

save(strcat("temp/",id,"-model.mat"))
end

function [optimal_parameter,self_predict_score]=get_parameter_train(X,Y)
col_X=size(X,2);
[sorted_corr,sorted_index]=feature_sort(X,Y);
sorted_corr=sorted_corr(sorted_corr>0.001);
sorted_index=sorted_index(sorted_corr>0.001);
X=X(:,sorted_index);
[optimal_parameter,self_predict_score]=GA_parameter(X,Y);
optimal_index=zeros(1,col_X);
optimal_index(:,sorted_index(logical(optimal_parameter)))=1;
optimal_parameter=optimal_index;
end

function [optimal_parameter,self_predict_score]=GA_parameter(X,Y)
CV=5;
col=size(X,2);
neigbor_num=5;
global global_X;
global global_Y;
global_X=X;
global_Y=Y;
global global_index;
rand('state',1);
global_index=crossvalind('Kfold',size(global_X,1), CV);
options=gaoptimset('PopulationSize',100,'Generations',60,'PopulationType', 'bitstring','TimeLimit',40000,'UseParallel','always','Display', 'iter');
[x,fval,exitflag,output,population,scores]=ga(@lbw,col,options);
[min_value,I]=min(scores);
optimal_parameter=population(I,:);
index=logical(optimal_parameter);
self_predict_score=zeros(size(global_Y,1),size(global_Y,2));

for k=1:CV
    test_index=logical(global_index==k);
    train_index=~(test_index);
    [Prior,PriorN,Cond,CondN]=mlknn_train(global_X(train_index,index),global_Y(train_index,:),neigbor_num,1);
    self_predict_score(test_index,:)=mlknn_test(global_X(train_index,index),global_Y(train_index,:),global_X(test_index,index),global_Y(test_index,:),neigbor_num,Prior,PriorN,Cond,CondN);
end
end

function f=lbw(index)
global global_X;
global global_Y;
global global_index;
f=AUPR_cross_validation(logical(index),global_X,global_Y,global_index);
end

function aupr=AUPR_cross_validation(index,X,Y,cv_index)

neigbor_num=5;
[row,col]=size(Y);
predict_score=zeros(row,col);
CV=5;

for k=1:CV
    test_index=logical(cv_index==k);
    train_index=~(test_index);
    train_X= X(train_index,index);
    test_X= X(test_index,index);
    train_Y=Y(train_index,:);
    test_Y=Y(test_index,:);
    [Prior,PriorN,Cond,CondN]=mlknn_train(train_X,train_Y,neigbor_num,1);
    predict_score(test_index,:)=mlknn_test(train_X,train_Y,test_X,test_Y,neigbor_num,Prior,PriorN,Cond,CondN);
end
aupr=-AUPR(Y(:), predict_score(:));
end

function aupr=AUPR(real,predict)

max_value=max(predict);
min_value=min(predict);
lp=50;
threshold=(min_value:(max_value-min_value)/(lp-1):max_value)';

threshold_num=length(threshold);

num=size(real,1);

tn=zeros(threshold_num,1);
tp=zeros(threshold_num,1);
fn=zeros(threshold_num,1);
fp=zeros(threshold_num,1);

for i=1:threshold_num
    tp(i,1)=sum(logical(predict>=threshold(i) & real==1));
    
    tn(i,1)=sum(logical(predict<threshold(i) & real==0));
    
    fp(i,1)=sum(logical(predict>=threshold(i) & real==0));
    
    fn(i,1)=sum(logical(predict<threshold(i) & real==1));
    
end


x=tp./(tp+fn);
y=tp./(tp+fp);

[x,index]=sort(x);
y=y(index,:);

aupr=0;
x(1,1)=0;
y(1,1)=1;
x(threshold_num+1,1)=1;
y(threshold_num+1,1)=0;
aupr=0.5*x(1)*(1+y(1));
for i=1:threshold_num
    aupr=aupr+(y(i)+y(i+1))*(x(i+1)-x(i))/2;
end
end

function [Prior,PriorN,Cond,CondN]=mlknn_train(train_data,train_target,Num,Smooth)
train_target=train_target';
train_data(train_data==0)=-1;

%MLKNN_train trains a multi-label k-nearest neighbor classifier
%
%    Syntax
%
%       [Prior,PriorN,Cond,CondN]=MLKNN_train(train_data,train_target,num_neighbor)
%
%    Description
%
%       KNNML_train takes,
%           train_data   - An MxN array, the ith instance of training instance is stored in train_data(i,:)
%           train_target - A QxM array, if the ith training instance belongs to the jth class, then train_target(j,i) equals +1, otherwise train_target(j,i) equals -1
%           Num          - Number of neighbors used in the k-nearest neighbor algorithm
%           Smooth       - Smoothing parameter
%      and returns,
%           Prior        - A Qx1 array, for the ith class Ci, the prior probability of P(Ci) is stored in Prior(i,1)
%           PriorN       - A Qx1 array, for the ith class Ci, the prior probability of P(~Ci) is stored in PriorN(i,1)
%           Cond         - A Qx(Num+1) array, for the ith class Ci, the probability of P(k|Ci) (0<=k<=Num) i.e. k nearest neighbors of an instance in Ci will belong to Ci , is stored in Cond(i,k+1)
%           CondN        - A Qx(Num+1) array, for the ith class Ci, the probability of P(k|~Ci) (0<=k<=Num) i.e. k nearest neighbors of an instance not in Ci will belong to Ci, is stored in CondN(i,k+1)

[num_class,num_training]=size(train_target);

temp=pdist(train_data);

dist_matrix=squareform(temp);

dist_matrix=dist_matrix+diag(realmax*ones(1,num_training));

Prior=(Smooth+sum(train_target==ones(num_class,num_training),2))/(Smooth*2+num_training);
PriorN=1-Prior;

[temp,index]=sort(dist_matrix,2);
Neighbors=index(:,1:Num);

temp_Ci=zeros(num_class,Num+1); %The number of instances belong to the ith class which have k nearest neighbors in Ci is stored in temp_Ci(i,k+1)
temp_NCi=zeros(num_class,Num+1); %The number of instances not belong to the ith class which have k nearest neighbors in Ci is stored in temp_NCi(i,k+1)
for i=1:num_training
    temp=zeros(1,num_class); %The number of the Num nearest neighbors of the ith instance which belong to the jth instance is stored in temp(1,j)
    neighbor_labels=[];
    for j=1:Num
        neighbor_labels=[neighbor_labels,train_target(:,Neighbors(i,j))];
    end

    temp=sum(neighbor_labels==ones(num_class,Num),2);
    temp=temp';
    
    for j=1:num_class
        if(train_target(j,i)==1)
            temp_Ci(j,temp(j)+1)=temp_Ci(j,temp(j)+1)+1;
        else
            temp_NCi(j,temp(j)+1)=temp_NCi(j,temp(j)+1)+1;
        end
    end
end
for i=1:num_class
        Cond(i,:)=(Smooth+temp_Ci(i,:))/(Smooth*(Num+1)+sum(temp_Ci(i,:)));
        CondN(i,:)=(Smooth+temp_NCi(i,:))/(Smooth*(Num+1)+sum(temp_NCi(i,:)));
end
end

function Outputs=mlknn_test(train_data,train_target,test_data,test_target,Num,Prior,PriorN,Cond,CondN)
train_target(train_target==0)=-1;
train_target=train_target';

test_target(test_target==0)=-1;
test_target=test_target';
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

[num_class,num_training]=size(train_target);
[num_class,num_testing]=size(test_target);
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

function [sorted_corr,I]=feature_sort(X,Y)
for i=1:size(X,2),
    corr(i) = getmutualinfo(X(:,i), Y);
end

[sorted_corr,I]=sort(corr,'descend');
end

function c =getmutualinfo(d, f)
num_class=size(f,2);
for i=1:num_class
    m(i)=MutualInformation(d, f(:,i));
end
c=max(m);
end
