function [ReconstructionError, OrthogonalityError] = ... 
    CompareQR(matrix, scheme)
    %% For a given matrix and a given scheme, get the Reconstruction Error 
    % and the Orthogonality Error. 
    [m, n] = size(matrix);
    [Q, R] = scheme(matrix); 
    ReconstructionError = norm(Q*R - matrix)/(m*n);
    OrthogonalityError  = norm(Q'*Q - eye(n))/(size(Q, 2)^2);
end