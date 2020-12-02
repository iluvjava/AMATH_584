function [EigenValues, EigenVectors, Flag] = OneShotRQItr(A, v0, maxItr, tol, p)
    % Rayleigh Quotients interations subroutines. 
    % p: 
    %   A orthonormal matrix that is going to represent the eigen spaces
    %   for all the found eigenvalues. 
    %   the p is going to be used as a projector for trimming of
    %   components that lies in the found eigen subspace. 

    switch nargin
        case 2
            maxItr = 100; 
            tol    = 1e-6;
            p = zeros(size(A));
        case 3
            tol    = 1e-6; 
            p = zeros(size(A)); 
        case 4
            % bruh
            p = zeros(size(A));
        case 5
            
    end
    
    import java.util.ArrayList;
    v            = v0./norm(v0);
    l            = v'*A*v;
    I            = eye(size(A));
   
    EigenValues  = ArrayList();
    EigenVectors = ArrayList();
    
    EigenValues.add(l); 
    EigenVectors.add(v);
    
    for II = 1: maxItr
       w = (A - l.*I)\v;
       w = (I - p*p')*w;  % subspace removal. 
       v = w./norm(w);
       l = v'*A*v;
       EigenValues.add(l);
       EigenVectors.add(v);
       if norm(RayleighQ(v)) < tol
           disp("Oneshot Inverse RQ Iterations converged to TOL. "); 
           Flag = 0;
           return;
       end
    end
    disp("Oneshot Inverse RQ Iterations failed to converge after maxitr.");
    
    Flag = 1;
    function R = RayleighQ(x)
        R = (A*x - (x'*A*x)*x);
    end
    
end