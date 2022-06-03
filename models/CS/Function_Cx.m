function [ Cx ] = Function_Cx( NN_MTX, INTER_MTX )
    % Cx(i,l) is C_xi^l from the paper, namely "the number of neighbors 
    % of x_i inducing the lth side effect"
    Cx = logical(NN_MTX) * INTER_MTX;
end