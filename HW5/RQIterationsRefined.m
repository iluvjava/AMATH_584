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
    
    Tol    = 1e-6; 
    MaxItr = 200;
    
    EigenValues  = cell(M, 1); 
    EigenVectors = cell(M, 1);
    P            = zeros(M, M);    % Found Eigen Vectors Matrix, orthogonal
    V            = randn(M, 1) + 1i.*randn(M, 1);  % Guessed Eigenvector. 
    V            = V./norm(V);
    I            = eye(M);
    
    for II = 1: M
        [Lambdas, EigenVecs, Flag] = OneShotRQItr(A, V, MaxItr, Tol);
        if Flag == 1
           error("Not converging for some reasons.") 
        else
           disp("EigenVal, EigenVec subroutine successful. ");
        end
        
        EigenVec = EigenVecs(:, 1);
        Lambda   = Lambdas(1);
        P(:, II) = (I - P*P')*EigenVec; % Augment our Eignespace. 
        
        EigenValues{II}  = Lambda;
        EigenVectors{II} = EigenVec;
        
        R = RayLeighQuotientGradient(V);
        
        if R < Tol
           break;  
        end
        V = RandomGuess();
        V = V./norm(V);
        
    end
    
    function v = RandomGuess()
        v = randn(M, 1) + 1i.*randn(M, 1);
        v = (I - P*P')*v;
        v = v./norm(v);
    end
    
    function R = RayLeighQuotientGradient(x)
       R = A*x  - ((x'*A*x)/(x'*x)).*x;
       R = norm(R);
    end
    
end