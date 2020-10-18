function [A_tilde] = RankReduce(U, S, V, r)
%  The function does a rank reduction on the SVD matrices of a given matrix.
%  
%  U, S, V:
%     These are the matrix gotten from the SVD decomposition where diagonal
%     elements of matrix S has to be singular value ranked in decreasing 
%     order of magnitudes. 
%  
%  r:
%     The number of top rank singular values you want to use to approximate
%     the given matrix with. 
%  Return: 
%     The low rank approximation of the matrix that got decomposed to 
    if r > min(size(U, 1), size(V, 2))
        error("Value of r is larger than the number of possible singular values")
    end
    U_tilde = U(:, 1:r);
    S_tilde = S(1:r, 1:r);
    V_tilde = V(:, 1:r);
    A_tilde = U_tilde*S_tilde*V_tilde.'; 
end

