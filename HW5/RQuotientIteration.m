function [EigenValues, EigenVectors, Lambda, EigenSpace]... 
    = RQuotientIteration(A, n)
    % 
    % It will iterate a fix number of time then it
    % it will return all the eigenvelues during iterations, and all the
    % matching eigenvectors. 
    % Inputs: 
    %   A: A matrix, Hermitian, with unique eigen values. 
    % Return: 
    %   EigenValues: 
    %       The I, J element is the the I th eigenvalue, after J th
    %       iterations. 
    %   EigenVectors: 
    %       The (I, :, J) is the vector, and it's the I th Eigenvector
    %       after J th iterations. 
    %   Lambda: 
    %       The Lambda matrix. 
    %   EigenSpace: 
    %       The matrix for from Eigen decomposition. 
    %   Note:
    %       All the return results are sorted in ascending order of abs
    %       magnitude. 
    
    [M, N] = size(A);
    if M ~= N
        error("Must be a square matrix");
    end
    
    EigenValues  = zeros(M, n);
    EigenVectors = zeros(M, M, n);
    P            = zeros(M, M);      % Found Eigen Vectors Matrix. 
    V            = RandomGuess();    % Guessed Eigenvector. 
    V            = V./norm(V);
    Lambda       = V'*A*V;
    I            = eye(M);
    
    for II1 = 1: M
        for II2 = 1: n
             V      = (A - Lambda.*I)\V;
             V      = V./norm(V);
             Lambda = V'*A*V;
             
             EigenValues(II1, II2)     = Lambda;
             EigenVectors(:, II1, II2) = V;
             R                         = RayLeighQuotientGradient(V);
             disp(strcat("Rayleigh Quotient: ", num2str(R)));
        end
        % New initial Guess and re-iterate it. 
        P(:, II1) = V;
        V      = (I - P*P')*RandomGuess();  % ortho projection 
        V      = V./norm(V);
        Lambda = V'*A*V;
        
    end
    
    if nargout == 4
       Lambda         = EigenValues(:, end);
       [Lambda, Idx]  = sort(Lambda, 'ComparisonMethod', 'abs');
       EigenSpace     = EigenVectors(:, :, end);
       EigenSpace2    = zeros(size(EigenSpace));
       EigenVectors2  = zeros(size(EigenVectors));
       for II = 1: length(Idx)
           EigenSpace2(:, II) = EigenSpace(:, Idx(II));
           EigenVectors2(:, II, :) = EigenVectors(:, Idx(II), :);
       end
       EigenSpace = EigenSpace2;
       Lambda = diag(Lambda);
       EigenVectors = EigenVectors2; 
    end
    
    function v = RandomGuess()
        v = randn(M, 1);
        if ~isequal(A, A')
           v = v + 1i.*randn(M, 1);
        end
    end

    function R = RayLeighQuotientGradient(x)
       R = A*x  - ((x'*A*x)/(x'*x)).*x;
       R = norm(R);
    end
end
