function [U, S, V] = Rsvd(A, k)
    % Given any matrix A, this is going to do a random sampling on the
    % matirx and then approximate the SVD for the original big matrix. It's
    % going to be partial, but it's going to be fast. 
    %   k: rank of the significant features. 
    %   A: The super huge matrix that we don't have time to compute. 
    
    [m, n] = size(A);
    
    % Stage A: 
    O = randn(n, k);
    Y = A*O; 
    [Q, ~] = qr(Y, 0);  % significant subspace. 
    
    % Stage B:
    B = Q'*A;
    [U, S, V] = svd(B, "econ");
    U         = Q*U;    % reproject. 
    
end

