function [ NN ] = NearestNeighbors( M, K, ExcludedRows)
    % K is the upper bound on number of neighbors (e.g. 5) 
    % for drug i, NN(i,:) represent similarity scores 
    % of i and its K nearest neighbors (other row entries are zeros)
    [m,n] = size(M);
    
    %eliminate diagonal from set of neighbors
    M(1:m+1:m*m) = 0;
    
    % sort rows in descending order
    [SH ind] = sort(M,2,'descend');
    NN = zeros(m,m);
    
    for i = 1:m
        j = 1;
        cntr = 1;
        while (j <= m & cntr <= K)
            % next largest element (in row i) is at position col
            col = ind(i,j);
            if ~any(col==ExcludedRows)
                NN(i,col) = M(i,col);
                cntr = cntr + 1;
            end
            j = j + 1;
        end
    end
end

