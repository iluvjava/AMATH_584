% Here we make an refined RQ iterations scheme, where it will overcome the
% problem of convergence for the plain RQ iterations. Fixing the convergene
% problem for non-symmetric real matrices. 

function [EigenValues, EigenVectors] = RQIterationsRefined(A)
    % Find the eigenvalues and eigenvectors for the matrix, and this
    % matirx can be non-symmetric. 
    % Starts with an initial guess that is small under the for the rayleigh
    % and it only iterates on the perpendicular subspace of the eigen space
    
    [M, N] = size(A);
    if M ~= N
        error("Must be a square matrix");
    end
    
    EigenValues  = zeros(M, n);      
    EigenVectors = zeros(M, M, n);
    P            = zeros(M, M);      % Found Eigen Vectors Matrix, orthogonal
    V            = RandomGuess();    % Guessed Eigenvector. 
    V            = V./norm(V);
    Lambda       = V'*A*V;
    I            = eye(M);
    
    
    
    
    
    
    
end