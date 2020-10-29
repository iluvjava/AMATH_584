function [orthoErrors, restructErrors] = ... 
    PerformenceSubroutine(matrices, schemes)
    % Given the generator of matrices, and 2 schemes that you want to
    % compare with, this function performs analysis on them and returns all
    % the errors. 
    % Schemes: 
    %   Cell arrays containing all the function handles. 
    % Matrices: 
    %   The matrices should be given as a cells java.until.arraylist. 
    % 
    % Return: 
    %   The Reconstruction Errors and the Orthogonality Errors.   
    N = matrices.size(); 
    S = length(schemes);
    orthoErrors = zeros(S , N); 
    restructErrors = zeros(S, N); 
    
    for I = 1: N 
        for J = 1:S
            scheme = schemes{J};
            [E1, E2] = ErrorGet(matrices.get(I - 1), scheme); 
            restructErrors(J, I) = E1; 
            orthoErrors(J, I) = E2; 
        end
    end
    
end

function [ReconstructionError, OrthogonalityError] = ... 
    ErrorGet(matrix, scheme)
    %% For a given matrix and a given scheme, get the Reconstruction Error 
    % and the Orthogonality Error. 
    [m, n] = size(matrix);
    [Q, R] = scheme(matrix); 
    ReconstructionError = norm(Q*R - matrix)/(m*n);
    OrthogonalityError  = norm(Q'*Q - eye(n))/(size(Q, 2)^2);
end
