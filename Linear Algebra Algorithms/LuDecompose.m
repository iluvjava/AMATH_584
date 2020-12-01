function [L, U, P] = LuDecompose(A)
    % This function performs a LU matrix decomposition with the pivoting.
    % This is the  for midterm of AMATH 584. 
    [m, n] = size(A); 
    if m ~= n
        error("Matrix must be squared. ")
    end
    U = A; L = eye(m); P = eye(m);

    for K = 1: n - 1
        [MaxEntry, I] = max(abs(U(K:end, K)));
        if MaxEntry < 1e-16
           error("Matrix Hardly Invertible");
        end
        I  = I + K - 1;
        [U(K, K:m), U(I, K:m)] = swap(U(K, K:m), U(I, K:m));
        if K >= 2
            [L(K, 1: K - 1), L(I, 1: K - 1)] = ... 
                swap(L(K, 1: K - 1), L(I, 1: K - 1));
        end
        
        [P(K, :), P(I, :)] = swap(P(K, :), P(I, :));
        
        for J = K + 1: m
           L(J, K) = U(J, K)/U(K, K); 
           U(J, K:m) = U(J, K:m) - L(J, K)*U(K, K: m);
        end
        
    end
    U = triu(U);
    function [a, b] = swap(b, a)
    end
end