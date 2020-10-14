clear all; close all; clc; 

%% Get files
% dirinfo = dir("yale-faces\yalefaces_cropped\CroppedYale\**");
% dirinfo([dirinfo.isdir]) = [];

dirinfo = dir("yale-faces\yalefaces_uncropped\subject*.*");

%% Meta Setting, create matrix
TRAINING_SET = 1: length(dirinfo);

% Get Matrices 
Matrices = cell(1, length(TRAINING_SET));
for I = TRAINING_SET
    % TheImage = imread(strcat(dirinfo(I).folder, "\", dirinfo(I).name), "pgm");
    TheImage = imread(strcat(dirinfo(I).folder, "\", dirinfo(I).name));
    ImageSize = size(TheImage);
    Matrices{I} = TheImage;
end
% imshow(Matrices{100})

%% 
% Put them into a big matrix 
ColumnDataMatrix = zeros(size(Matrices{1}, 1)*size(Matrices{1}, 2), length(Matrices));
Column = 1;
for Matrix = Matrices
    Matrix = Matrix{1};
    ColumnDataMatrix(:, Column) = ... 
        reshape(Matrix, [size(Matrix, 1)*size(Matrix, 2), 1]);
    Column = Column + 1;
end
clearvars -except ColumnDataMatrix ImageSize;

%% 
% Time for Maths. 

ImageTotalDataPoints = size(ColumnDataMatrix, 1);
NumberofImages = size(ColumnDataMatrix, 2);
TotalAverage = (ColumnDataMatrix*ones(NumberofImages, 1))/NumberofImages;

figure; hold on; image(reshape(TotalAverage, [ImageSize(1) ImageSize(2)]));
title("Your Average Matrix Creepy Face");

[U, S, V] = svd(ColumnDataMatrix - TotalAverage, 'econ');  % SVD on Variance Matrix! 

%% Look at the singular values 

figure;
subplot(2, 1, 1);
plot(1:length(diag(S)), log(diag(S)), '-o');
title("All of The Singular Values");
ylabel("Log(\sigma_i)")

subplot(2, 1, 2);
SingularVals = log(diag(S));
plot(1:100, SingularVals(1:100), '-o');
title("the first 200 Singular values");
ylabel("Log(\sigma_i)")

%% Look at the Basis in U
figure;
title("first 16 Basis in U (EigenFaces)");
for I = 1:16
   subplot(4, 4, I);
   ImgArr = U(:, I);
   imshow(ArrayToGrayScale(ImgArr, ImageSize));
end

%% reconstructions For Known Faces 
% define good constants: 
NUMBER_OF_RANDOM_FACES = 3;
RECONSTRUCTION_RANK = 200;

% Reconstruct for faces inside of the known data set. 
U_tild = U;
S_tild = S;
S_tild(RECONSTRUCTION_RANK + 1: end, :) = 0; 
V_tild = V;  % Careful about here, because USV^T
A_tild = U_tild*S_tild*V_tild.';
A_tild = A_tild + TotalAverage;

RandomFaces = ...
    randi([1 size(ColumnDataMatrix, 2)], NUMBER_OF_RANDOM_FACES, 1); 
figure;
for I = 1: NUMBER_OF_RANDOM_FACES
    FaceID  = RandomFaces(I);
    TheFace = ColumnDataMatrix(:, FaceID); 
    subplot(2, NUMBER_OF_RANDOM_FACES, I);
    imshow(ArrayToGrayScale(TheFace, ImageSize));
    TheFaceReconstruct = A_tild(:, FaceID);
    subplot(2, NUMBER_OF_RANDOM_FACES, NUMBER_OF_RANDOM_FACES + I);
    imshow(ArrayToGrayScale(TheFaceReconstruct, ImageSize));
end
%% Variance Analysis
% Instead of plotting it, I am going to visualize this numerically to 
% See the errors for low rank approximation. 
% Technique: Variance Analysis.

RANKS = 1:10:min(size(ColumnDataMatrix, 2), 200);
VarianceUnexplained = zeros(length(RANKS));
TotalVariance = (ColumnDataMatrix - TotalAverage);
TotalVariance = ...
    reshape(TotalVariance, [1, size(TotalVariance, 1)*size(TotalVariance, 2)]);
TotalVariance = var(TotalVariance); 

Variances = [];
for R = RANKS  % This part can be made faster with dynamic programming, but whatever.
    A_tilde = RankReduce(U, S, V, R);
    A_tilde = reshape(A_tilde, [1, size(A_tilde, 1)*size(A_tilde, 2)]);
    VarianceExplained = var(A_tilde);
    Variances(end + 1) = VarianceExplained/TotalVariance;
end
figure;
plot(RANKS, Variances, "-x");
title("Ranks and Explained Variances");
disp("The threshold of ranks that gives above 95% explain variance is approximately: r = 71");








