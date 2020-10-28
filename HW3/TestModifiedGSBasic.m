clc; clear variables;

Trials = 1; M = 1000, N = 500;

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

%%

BadMatrix = vander(1:25);
[Q, R] = qr(BadMatrix);
TheError = norm(Q*R - BadMatrix);

% RandMatrix = rand(10);
% [Q, R] = ModifiedGS(RandMatrix);
% norm(Q*R - RandMatrix)

