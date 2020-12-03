% Let's try to do HW4.


%% Part (a), (b)
clear variables; clc;

A = RandomSymmetricMatrixNormal(10);
[U, V] = eigs(A, 2, "largestabs");
[EigenValue, EigenVector] = PowerIterate(A, 200);

Vec1 = abs(U(:, 1));

Errors = Vec1 - abs(EigenVector); % Absolute value needed.
ErrorsNorm = zeros(1, size(Errors, 2));
for II = 1: size(Errors, 2)
    ErrorsNorm(II) = norm(Errors(:, II));
end
xs = 1: length(ErrorsNorm);
figure; 
semilogy(xs, ErrorsNorm); hold on;
semilogy(xs, (abs(V(2,2)/V(1,1))).^xs, 'k-.');
title("Power Iteration, Rate of Convergence (Symmetric)");
ylabel("Errors");
xlabel("Iterations");
legend("|v_i - v_{Truth}|", "|\lambda_1/\lambda_2|");
saveas(gcf, "pitrsym", "png");

%% Part (c)
clear variables;
A = RandomSymmetricMatrixNormal(10);
n = 10;
[EigenValue, EigenVector, Lambda, EigenSpace] = RQuotientIteration(A, n);
[U, V] = eigs(A, 10, "smallestabs");

%% 
% Plot, convergence of the eigen vectors. 
figure;
for II = 1: 10     % II th vector.
    Errors = zeros(1, n);
    for JJ = 1: n  % JJ th iterations. 
        EigenVec = EigenVector(:, II, JJ);
        EigenVec = reshape(abs(EigenVec), 10, 1);
        Errors(JJ) = norm(EigenVec - abs(U(:, II)));
    end
    semilogy(1: length(Errors), Errors); hold on;
    semilogy(1: length(Errors), Errors, "o-");
end
title("Rayleigh Quotient Iterations");
xlabel("Number of Iterations");
ylabel("Log(Errors)");
saveas(gcf, "rayleigh-itr-sym", "png");

%% Part (d)
% Repeat the exact same thing but for non symmetric matrices. 

clear variables; clc;

A = randn(10);
[U, V]           = eigs(A, 2, "largestabs");
[~, EigenVector] = PowerIterate(A, 200);

Vec1 = abs(U(:, 1));

Errors     = Vec1 - abs(EigenVector); % Absolute value needed.
ErrorsNorm = zeros(1, size(Errors, 2));
for II = 1: size(Errors, 2)
    ErrorsNorm(II) = norm(Errors(:, II));
end
figure;
xs = 1: length(ErrorsNorm);
semilogy(xs, ErrorsNorm); hold on;
semilogy(xs, (abs(V(2,2)/V(1,1))).^xs, 'k-.');
title("Power Iteration, Rate of Convergence (Non-Symmetric)");
ylabel("Errors");
xlabel("Iterations");
legend("|v_i - v_{Truth}|", "|\lambda_1/\lambda_2|");
saveas(gcf, "p-itr-nonsym", "png");

%% 
clear variables; clc;
A = randn(10);
n = 20;
[EigenValue, EigenVector, Lambda, EigenSpace] = RQuotientIteration(A, n);
[U, V] = eigs(A, 10, "smallestabs");

%% 
% Plot, convergence of the eigen vectors. 
figure;
for II = 1: 10     % II th vector.
    Errors = zeros(1, n);
    for JJ = 1: n  % JJ th iterations. 
        EigenVec    = EigenVector(:, II, JJ);
        EigenVec    = arrayfun(@abs, EigenVec);
        GroundTruth = arrayfun(@abs, U(:, II));
        Errors(JJ)  = norm(EigenVec - GroundTruth);
    end
    semilogy(1: length(Errors), Errors); hold on;
    semilogy(1: length(Errors), Errors, "o-");
end
title("Rayleigh Quotient Iterations(Non-Symmetric)");
xlabel("Number of Iterations");
ylabel("Log(Errors)");
saveas(gcf, "ray-itr-nonsym", "png");

%% INTVESTIGATIONS! RANDOM GUESS
% Here are are going explore a bit about the Rayleigh Quotients and their
% convergence properties. 
clear variables; clc; close all;
M      = randn(10); 
[U, V] = eigs(M, 10, "largestabs");
figure;
for I = 1: 300
    v0     = randn(10, 1) + 1i*randn(10, 1);
    [EigenValue, EigenVector, Flag] = OneShotRQItr(M, v0);
    plot(real(EigenValue), imag(EigenValue), 'o-'); hold on;
end

% Plot eigen vectors. 
EigenValues = diag(V);
plot(real(EigenValues), imag(EigenValues), 'kx', "markersize", 12, "linewidth", 2);
title("Eigenvalues, Inverse Iterations, RandomGuesses");
xlabel("Re")
ylabel("im")
saveas(gcf, "p-itr-nonsym-randomguesses", "png");

%% SMART GUESSES
clear variables; clc; close all;
M      = randn(10); 
[U, V] = eigs(M, 10, "largestabs");
figure;
for II = 1: 10
    v0 = randn(10, 1) + 1i*randn(10, 1);
    v0 = 0.05*v0 + U(:, II);
    [EigenValue, EigenVector, Flag] = OneShotRQItr(M, v0);
    plot(real(EigenValue), imag(EigenValue), 'o-'); hold on;
end

% Plot eigen vectors. 
EigenValues = diag(V);
plot(real(EigenValues), imag(EigenValues), 'kx', "markersize", 12, "linewidth", 1);
title("Reyleigh Q Smart Guesses");
saveas(gcf, "p-itr-nonsym-smartguesses", "png");

%% MORE INVESTIGATIONS
% clear variables; clc; close all;
% M      = randn(10); 
% [U, V] = eigs(M, 10, "largestabs");
% v0     = randn(10, 1) + 1i*randn(10, 1);
% [EigenValues, EigenVectors] = RQIterationsRefined(M);

%% 
function M = RandomSymmetricMatrixNormal(n)
    % I adhere to make a random Symmetric matrix that has elements normally
    % distributed, so we can have both positve and negative values. 
    % The word: Symmetric implies that the matrix is going to be real. 
    M = randn(n, n);
    M = triu(M) + tril(M');
end

