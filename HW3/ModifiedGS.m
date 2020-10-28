function [Q, R] = ModifiedGS(A)
    % Given a matrix A, this function produces a reduced Gram Schitmz for
    % the matrix, using the Modified Gram Schimtz process. 
    [m, n] = size(A);
    Q = A;  % Copy
    Q(:, 1) = Q(:, 1)/norm(Q(:, 1));
    for I = 2: n
        q = Q(:, I - 1);
        if norm(Q) < 1e-15
           error("Rank Deficit matrix can't use Modified GS. ") ;
        end
        P = q*q'; 
        V =  Q(:, I: end);
        Q(:, I: end) = V - P*V;
        q = Q(:, I);
        Q(:, I) = q/norm(q);
    end
    R = triu(Q'*A);
end