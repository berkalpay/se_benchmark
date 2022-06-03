function []=predict(id, J, rnk, iter, lR, lM, lN)

    J = str2double(J);
    rnk = str2double(rnk);
    iter = str2double(iter);
    lR = str2double(lR);
    lM = str2double(lM);
    lN = str2double(lN);

    train_X=dlmread(strcat("temp/",id,"-fit_X.csv"), "\t", 1, 1);
    test_X=dlmread(strcat("temp/",id,"-predict_X.csv"), "\t", 1, 1);
    X=[train_X; test_X];
    train_y=dlmread(strcat("temp/",id,"-fit_y.csv"), "\t", 1, 1);
    test_y = zeros(size(test_X,1),size(train_y,2));

    % specify the input matrices
    R = [train_y; test_y]; % SIDER associatiowithout postmarketing
    M = TanimotoCoeff(X); % Tanimoto similarity
    N = dlmread(strcat("temp/",id,"-ADR_similarity.csv"), "\t", 1, 1); % UMLS similarity of ADRs
    
    % set degree matrices to be used in the CS method
    [DM nM]= GetDiag(M,J);
    [DN, nN]= GetDiag(N,J);
    DMM = DM-nM;
    DNN = DN-nN;
    
    [m,n] = size(R);
    
    TEST_MTX = R;

    % our own Compressed Sensing method (CS)
    indices = sub2ind(size(R), repelem((size(train_y,1)+1):size(R,1),size(R,2)), repmat(1:size(R,2),1,size(test_y,1)));
    MULT_LAB = MLKNN_TEST(R, M, indices, J, 1);
    raw_normalization = 0;
    [ unimportant, ExcludedColumns ] = find(sum(TEST_MTX,1)==0); % this should be a constant
    [ ExcludedRows unimportant ] = find(sum(TEST_MTX,2)==0); % this should be given?
    W = max(1, 6 * TEST_MTX);
    IMPUTE = zeros(m,n);
    [F G] = WeightImputeLogFactorization(TEST_MTX,DMM,DNN,W,IMPUTE,lR,lM,lN,iter,rnk);
    [F G] = WeightedProfile(F, G, M, N, ExcludedRows, ExcludedColumns, J, raw_normalization);
    EXC_COS = GetP(F*G');
    EXC_COS(ExcludedRows,:) = EXC_COS(ExcludedRows,:) .* MULT_LAB(ExcludedRows,:);
    
    PREDMTX_COS = EXC_COS((end-size(test_X,1)+1:end),:);
    dlmwrite(strcat("temp/",id,"-predict_y.csv"), PREDMTX_COS, "\t")
        
end
