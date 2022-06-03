% postprocess all empty rows and columns (cold-start)
function [ nF nG ] = WeightedProfile(F, G, M, N, ExcludedRows, ExcludedColumns, J, raw_normalization)

    % if raw normalization is 1, then sum is incremented by 1; otherwise by
    % score (Mx)
    
    m = size(M,1);
    n = size(N,1);
           
    % weight placed on the original (input) latent matrices F and G
    % computed by the WeightedImputeLogFactorization
    WP = 0.0001;
    
    nF = F;
    for i = 1:m
       % make sure the row is empty (all zeros)
       if sum(any(i==ExcludedRows))>0
            Row = M(i,:);       
            Row(ExcludedRows) = -1 * realmax;
            summation = 0;
            nF(i,:) = WP * F(i,:);
            for j=1:J
                % find the next most similar row (drug)
               [Mx Ix] = max(Row);
               if raw_normalization == 1
                    summation = summation + 1;
               else
                   summation = summation + Mx;
               end
               nF(i,:) = nF(i,:) + Mx * F(Ix,:);
               Row(Ix) = -1 * realmax;
            end
            % normalize
            nF(i,:) = nF(i,:) / (WP + summation);
       end
       
    end
    
    nG = G;
    for i = 1:n
       % make sure the column (ADR) is all zeros
       if sum(any(i==ExcludedColumns))>0
            Col = N(i,:);
            Col(ExcludedColumns) = -1 * realmax;
            summation = 0;
            nG(i,:) = WP * G(i,:);
            for j=1:J
                % find the next most similar ADR
                [Nx Ix] = max(Col);
                if raw_normalization == 1
                    summation = summation + 1;
                else
                   summation = summation + Nx;
                end
                nG(i,:) = nG(i,:) + Nx * G(Ix,:);
                Col(Ix) = -1 * realmax;
            end
            % normalize
            nG(i,:) = nG(i,:) / (WP + summation); 
       end
    end 
end
