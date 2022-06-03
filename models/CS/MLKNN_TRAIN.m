% Compute prior PH0, PH1 and posterior probabilities PEH0, PEH1
% (see page 4 of the above reference for details)

%     INPUT:
% "R" is the mxn matrix of drug-ADR associations (e.g. SIDER)
% "M" is the mxm matrix of pairwise drug similarity scores (e.g. Tanimoto)
% "Excluded" represents the set of indices in R to mask-out (hide and then
% "rediscover/predict")
% "K" is the number of nearest neighbors
% "smooth" is the smooth factor (see page 4 of the above reference) 

%     OUTPUT:
% PH0, PH1 - prior probabilities 
% PEH0, PEH1 - posterior conditional probabilities 

function [ PH0, PH1, PEH0, PEH1 ] = MLKNN_TRAIN(R, M, Excluded, K, smooth)
    % excluded is a logical of same dimension as R, 
    % where 1 indicates excluded element
    TRAIN_R = R;
    TRAIN_R(Excluded) = 0;
    
    [ExcludedRows unimportant] = find(sum(TRAIN_R,2)==0);
    [unimportant, ExcludedColumns] = find(sum(TRAIN_R,1)==0);
    
    TRAIN_R(ExcludedRows, :) = [];
    
    TRAIN_M = M;
    TRAIN_M(ExcludedRows, :) = [];
    TRAIN_M(:, ExcludedRows) = [];
    
    % PH1 is n-dimensional vector (one component per side effect)
    % PH1(l) is "the event that a drug has l-th side effect"
    % smooth is the smoothing factor (default is 1)
    
    [PH0, PH1] = Function_PH( TRAIN_R, [], smooth );
    
    % TRAIN_NN matrix is obtained from TRAIN_M by keeping the highest K 
    % scores in each row and replacing remaining (non-neighbor) scores by 0
    TRAIN_NN = NearestNeighbors( TRAIN_M, K, [] );
    
    % compute number of neighbors inducing side effects
    TRAIN_Cx = Function_Cx( TRAIN_NN, TRAIN_R );
    
    % Note: Because indices in MATLAB start from 1, the entry at position 
    % (1,l) in fact represents c(0,l)
    [ c ] = Function_c( TRAIN_R, TRAIN_Cx, K );
    [ c_prime ] = Function_c( 1 - TRAIN_R, TRAIN_Cx, K );
    
    [ PEH1] = Function_ConditionalPE ( c, K, smooth );
    [ PEH0 ] = Function_ConditionalPE ( c_prime, K, smooth );
end

function [ PH0, PH1 ] = Function_PH( INTER_MTX, ExcludedRows, smooth )
    % PH1 is n-dim vector where n is the number of side effects
    % PH1(l) is P(H_1^l) from 's paper i.e., 
    % "the event that a drug has l-th side effect"
    % More specifically, it is a vector of side effect frequencies
    % smooth is the smoothing factor (default is 1)
    INTER_MTX(ExcludedRows, :) = [];
    [m, n] = size(INTER_MTX);
    PH1 = (smooth + sum(INTER_MTX,1)) / (2 * smooth + m);
    PH0 = 1 - PH1;
end

function [ c ] = Function_c(TRAIN_R, TRAIN_Cx, K )
    % Note 1: because indices in MATLAB start from 1, we adjust our array
    % accordingly (keep reading ... )
    
    % K is the user selected upper bound on number of neighbors (e.g. 5) 
    
    % c(j,l) (which is c_l[j-1] from the paper), represents "the number of 
    % instances with side effect l which have exactly j-1 neighbors with 
    % side effect l in their nearest neighbors". In other words, c(j,l) is 
    % the number of instances i such that TRAIN_MTX(i,l) = 1 and Cx(i,l) = j-1
    
    % Note 2: input matrix C is computed by the function 
    % Cx( NN_MTX, INTER_MTX )
    
%     for j=1:K + 1
%         c(j, :) = sum((1- logical(Cx - (j-1))),1);
%     end
    
    % c_prime(j, l) is the number of instances i without side effect l
    % which have exactly j-1 neighbors with l-th side effect. In other 
    % words, c_prime(j,l) is the number of instances i such that
    % TRAIN_MTX(i,l) = 0 and Cx(i,l) = j-1.

    [m, n] = size(TRAIN_Cx);
    c = zeros( K+1, n);
    for i=1:K+1
       c(i,:) = sum( TRAIN_R .* ( 1 - logical( TRAIN_Cx - (i -1) )), 1 ); 
    end
    
    % Note: to compute c_prime, use 1-TRAIN_MTX instead of TRAIN_MTX as input
end

% for PEH1 use c; for PEH0 use c_prime as input
function [ PEH ] = Function_ConditionalPE( c, K, smooth )
    % Again, indices are off by 1
    
    % check whether K = (# rows in c) -1
    
    [x, y] = size(c);
    if ~(x == (K + 1)) 
        msg = 'Error occurred in Function_ConditionalPE';
        error(msg)
    end
    
    Norm = smooth * (K + 1) + sum(c, 1);
    % for efficient division, make a matrix with all rows equal to Norm
    ONE = ones(K+1, 1);
    NORM_MTX = ONE * Norm;
    
    PEH = (smooth + c) ./ NORM_MTX;
end
