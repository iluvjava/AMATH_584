function [Bestrank, CulmulativeVariance] = ...
    VarianceAnalysis(u, s, v, originalMatrix, threshold)
%   Using the explained variance of the reconstructed matrix to determine
%   the best number of singular values needed to reconstruct the matrix. 
%   u, s, v: 
%       The matrices decomposed from the SVD decomposition. 
%   originalMatrix: 
%       The matrix that got decomposed into u, s, v
%   threshold:
%       threshold for explained variance
    ReconstructedMatrix = zeros(size(u, 1), size(v, 1));
    CulmulativeVariance = [];
    for I = 1: size(s, 1)
        ReconstructedMatrix = ...
            ReconstructedMatrix + u(:, I)*s(I, I)*v(:, I).';
        CulmulativeVariance(I) = getVarianceFor(ReconstructedMatrix);
        CulmulativeVariance(I) = ... 
            CulmulativeVariance(I)/getVarianceFor(originalMatrix);
        if CulmulativeVariance(I) > threshold
            Bestrank = I;
            break;
        end
    end
  
    function MatrixVariance = getVarianceFor(matrix)
        MatrixVariance = ...
            var(reshape(matrix, [1, size(matrix, 1)*size(matrix, 2)]));
    end
end