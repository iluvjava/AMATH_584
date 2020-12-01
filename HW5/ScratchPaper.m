% Let's try to do HW4.


%% Part (a), (b)
clear variables; clc;

A = RandomSymmetricMatrixNormal(10);
[U, V] = eigs(A, 2, "largestabs");
[EigenValues, EigenVectors] = PowerIterate(A, 200);

Vec1 = abs(U(:, 1));

Errors = Vec1 - abs(EigenVectors); % Absolute value needed.
ErrorsNorm = zeros(1, size(Errors, 2));
for II = 1: size(Errors, 2)
    ErrorsNorm(II) = norm(Errors(:, II));
end
xs = 1: length(ErrorsNorm);
semilogy(xs, ErrorsNorm); hold on;
semilogy(xs, (abs(V(2,2)/V(1,1))).^xs, 'k-.');
title("Power Iteration, Rate of Convergence (Symmetric)");

%% Part (c)
clear variables;
A = RandomSymmetricMatrixNormal(10);
n = 10;
[EigenValues, EigenVectors, Lambda, EigenSpace] = RQuotientIteration(A, n);
[U, V] = eigs(A, 10, "smallestabs");

%% 
% Plot, convergence of the eigen vectors. 
figure;
for II = 1: 10     % II th vector.
    Errors = zeros(1, n);
    for JJ = 1: n  % JJ th iterations. 
        EigenVec = EigenVectors(:, II, JJ);
        EigenVec = reshape(abs(EigenVec), 10, 1);
        Errors(JJ) = norm(EigenVec - abs(U(:, II)));
    end
    semilogy(1: length(Errors), Errors); hold on;
    semilogy(1: length(Errors), Errors, "o-");
end
title("Rayleigh Quotient Iterations");
xlabel("Number of Iterations");
ylabel("Log(Errors)");

%% Part (d)
% Repeat the exact same thing but for non symmetric matrices. 

clear variables; clc;

A = randn(10);
[U, V] = eigs(A, 2, "largestabs");
[EigenValues, EigenVectors] = PowerIterate(A, 200);

Vec1 = abs(U(:, 1));

Errors = Vec1 - abs(EigenVectors); % Absolute value needed.
ErrorsNorm = zeros(1, size(Errors, 2));
for II = 1: size(Errors, 2)
    ErrorsNorm(II) = norm(Errors(:, II));
end
xs = 1: length(ErrorsNorm);
semilogy(xs, ErrorsNorm); hold on;
semilogy(xs, (abs(V(2,2)/V(1,1))).^xs, 'k-.');
title("Power Iteration, Rate of Convergence (Non-Symmetric)");

%% 
clear variables; clc;
A = randn(10);
n = 10;
[EigenValues, EigenVectors, Lambda, EigenSpace] = RQuotientIteration(A, n);
[U, V] = eigs(A, 10, "smallestabs");

%% 
% Plot, convergence of the eigen vectors. 
figure;
for II = 1: 10     % II th vector.
    Errors = zeros(1, n);
    for JJ = 1: n  % JJ th iterations. 
        EigenVec   = EigenVectors(:, II, JJ);
        EigenVec   = reshape(abs(EigenVec), 10, 1);
        Errors(JJ) = norm(EigenVec - abs(U(:, II)));
    end
    semilogy(1: length(Errors), Errors); hold on;
    semilogy(1: length(Errors), Errors, "o-");
end
title("Rayleigh Quotient Iterations(Non-Symmetric)");
xlabel("Number of Iterations");
ylabel("Log(Errors)");


%% 
function M = RandomSymmetricMatrixNormal(n)
    % I adhere to make a random Symmetric matrix that has elements normally
    % distributed, so we can have both positve and negative values. 
    % The word: Symmetric implies that the matrix is going to be real. 
    M = randn(n, n);
    M = triu(M) + tril(M');
end

