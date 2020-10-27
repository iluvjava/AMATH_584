function [Q, R] = ModifiedGS(A)
    % Given a matrix A, this function produces a reduced Gram Schitmz for
    % the matrix, using the Modified Gram Schimtz process. 
    [m, n] = size(A);
    Q = A;  % Copy
    for I = 1: n
        q = Q(:, I);
        q = q/norm(q);
        Q(:, I) = q;
        P = q*q.';
        Q(:, I + 1: end) = Q(:, I + 1: end) - P*Q(:, I + 1: end);
    end
    R = Q.'*A;
end