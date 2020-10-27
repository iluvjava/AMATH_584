clc; clear variables;

Trials = 10; M = 1000, N = 500;

ReconstructionError = zeros(1, Trials);
IdentityError = zeros(1, Trials);
IdentityMatrix = eye(N);
for Trial = 1: Trials
   RandMatrix = rand(M, N);
   [Q, R] = ModifiedGS(RandMatrix);
   ReconstructionError(Trial) = norm(Q*R - RandMatrix);
   IdentityError(Trial) = norm(Q.'*Q - IdentityMatrix);
end

ReconstructionError
IdentityError