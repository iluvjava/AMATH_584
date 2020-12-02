function [EigenValues, EigenVectors]= PowerIterate(A, n)
    % Use the poewr iterate method to look for the Dominanting Eigenvalues
    % in a Hermitian Matrix. 
    % 
    % The function will: 
    % 
    %     Faithfually iterate "n" times and then return the dominanting
    %     eigenvalues and eigen vector. 
    %       
    %     Stores all the eigenvalues and eigen vectors into a matrix and
    %     a vector. 
    %       
    %     If Matrix is not symmetric, it will initialize the vector with a
    %     complex numbers in it. 
    % A: 
    %   HERMITIAN with unique eigen values!!!! 
    
    [M, N] = size(A);
    if M ~= N
        error("Must be a square matrix. ")
    end
    
    EigenValues  = zeros(1, n);
    EigenVectors = zeros(M, n);
    
    v = randn(M, 1);
    if ~isequal(A, A.')
        v = v + 1i.*rand(M, 1);
    end
    v   = v./norm(v);
    for Itr = 1: n
        w = A*v;
        v = w./norm(w);
        lambda = v'*A*v;
        EigenValues(Itr) = lambda;
        EigenVectors(:, Itr) = v;
        disp(strcat("Rayleigh Quotient: ", num2str(RayLeighQuotientGradient(v))));
    end
    
    function R = RayLeighQuotientGradient(x)
       R = A*x  - ((x'*A*x)/(x'*x)).*x;
       R = norm(R);
    end
end