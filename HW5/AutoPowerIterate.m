function [EigenVectors, EigenValues] = AutoPowerIterate(A)
    % Does a Automatic Power iterations on the given matrix, with a random
    % initial guess. 
    % Rayleigh Quotient is going to be the stopping conditions
    
    [M, N] = size(A);
    TOL    = 1e-4;
    if M ~= N
        error("Must be a square matrix. ")
    end
    
    EigenValues  = 0;
    EigenVectors = zeros(M, 1);
    
    v = randn(M, 1);
    if ~isequal(A, A.')
        v = v + 1i.*rand(M, 1);
    end
    v   = v./norm(v);
    Itr = 0;
    while 1
        Itr = Itr + 1;
        w   = A*v;
        v   = w./norm(w);
        lambda                   = v'*A*v;
        EigenValues(end + 1)     = lambda;
        EigenVectors(:, end + 1) = v;
        R = RayLeighQuotientGradient(v);
        disp(strcat("Rayleigh Quotient: ", num2str(R)));
        
        if R < TOL
            disp(strcat("Break with Itr: ", num2str(Itr)));
            break;
        end
    end
    
    function R = RayLeighQuotientGradient(x)
       R = A*x  - ((x'*A*x)/(x'*x)).*x;
       R = norm(R);
    end
end

