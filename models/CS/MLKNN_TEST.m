% Performs "multi-label k-nearest neighbor' as described in 
% Zhang et al., BMC Bioinformatics (2015) 16:365

%     INPUT:
% "R" is the mxn matrix of drug-ADR associations (e.g. SIDER)
% "M" is the mxm matrix of pairwise drug similarity scores (e.g. Tanimoto)
% "Excluded" represents the set of indices in R to mask-out (hide and then
% "rediscover/predict")
% "K" is the number of nearest neighbors
% "smooth" is the smooth factor (see page 4 of the above reference) 

%     OUTPUT:
% "EXC" is the reconstructed matrix R with masked out entries ("Excluded") 
% filled out by values predicted by ML algorithm


function [ EXC ] = MLKNN_TEST( R, M, Excluded, K, smooth )
    % compute prior PH0, PH1 and posterior probabilities PEH0, PEH1 
    [ PH0, PH1, PEH0, PEH1 ] = MLKNN_TRAIN (R, M, Excluded, K, smooth);    
    NN_M = NearestNeighbors( M, K, [] );
    TEST_MTX = R;
    TEST_MTX(Excluded) = 0;
    Cx = Function_Cx( NN_M, TEST_MTX );    
    
    EXC = R;
    
    for k=1:length(Excluded)
        [r, l] = ind2sub(size(TEST_MTX),Excluded(k));
        has_se = PH1(l) * PEH1(Cx(r,l) + 1, l);
        hasnot_se = PH0(l) * PEH0(Cx(r,l) + 1 , l);
        tot = has_se + hasnot_se;
        if tot > 0
            EXC(r,l) = has_se / tot;
        else
            EXC(r,l) = PH1(l);
        end
   end
end

