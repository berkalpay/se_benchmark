function [ T ] = TanimotoCoeff( CFEAT )
    [m n] = size(CFEAT);
    T = zeros(m,m);
    for i=1:m
        for j=1:m
            x = CFEAT(i,:);
            y = CFEAT(j,:);
            num = sum(x & y);
            denom = sum(x|y);
            if denom > 0
                T(i,j) = num / denom;
            else
                T(i,j) = 0;
            end
        end
    end
end
