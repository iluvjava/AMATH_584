clear all; close all; clc; 

%% Get files
dirinfo = dir("yale-faces\yalefaces_cropped\CroppedYale\**");
dirinfo([dirinfo.isdir]) = [];

% dirinfo = dir("yale-faces\yalefaces_uncropped\subject*.*");

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

%% PLOTING THE SINGULAR VALUES RELATED STUFF. 

figure;
subplot(2, 1, 1);
plot(1:length(diag(S)), log(diag(S)), '-o');
title("All of The Singular Values");
ylabel("Log(\sigma_i)")

subplot(2, 1, 2);
SingularVals = log(diag(S));
plot(1:100, SingularVals(1:100), '-o');
title("the first 100 Singular values");
ylabel("Log(\sigma_i)")
saveas(gcf, "Singular_Value_Distribution.png");

figure;
histogram(diag(S));
title("Histogram of Singular Value");
saveas(gcf, "Singular_Value_Histogram.png");

figure; 
histogram(log(diag(S)));
title("Histogram of $$\log{\sigma_i}$$", 'Interpreter','latex');
saveas(gcf, "Singular_Value_logrithm_histogram.png");

%% LOOKING A EIGEN FACES AND PLOTING IT OUT. 
figure;
title("first 16 Basis in U (EigenFaces)");
for I = 1:16
   subplot(4, 4, I);
   ImgArr = U(:, I);
   imshow(ArrayToGrayScale(ImgArr, ImageSize));
end

saveas(gcf, "EigenFaces.png")

%% Variance Analysis 
% COMPUTATIONALLY HEAVY
% Instead of plotting it, I am going to visualize this numerically to 
% See the errors for low rank approximation. 
% Technique: Variance Analysis.

[Bestrank1, CulmulativeVariance] = ... 
    VarianceAnalysis(U, S, V, ColumnDataMatrix - TotalAverage, 0.95);

%% VARIANCE ANALYSIS AND PLOTING.
figure;
plot(CulmulativeVariance, "ko");
ylabel("Explained Squared Variance");
title("Variance Analysis");
xline(Bestrank1);
saveas(gcf, "Rank and Explained Squared Variance.png");



%% Energy Analysis AND PLOTING.
% Another way of determing the rank of the matrix is to use the idea of
% energy, see code 1.19 in the data book for more details.
Bestrank2 = EnergyAnalysis(S);
saveas(gcf, "Energy Analysis.png");
Bestrank90 = EnergyAnalysis(S, 0.9);
Bestrank99 = EnergyAnalysis(S, 0.99);

%% Reconstruction of Known Faces 
% Using the new gotten ranks for reconstructing a certain column in the
% original matrix. 
ReconstrctRandomFaces... 
    (ColumnDataMatrix, U, S, V, Bestrank1, ImageSize, TotalAverage);
saveas(gcf, "Random Faces Reconstruction using Variance Analysis.png");

ReconstrctRandomFaces... 
    (ColumnDataMatrix, U, S, V, Bestrank2, ImageSize, TotalAverage);
saveas(gcf, "Random Faces Reconstruction using Energy Analysis.png")

function ReconstrctRandomFaces(dataMatrix, U, S, V, rank, ImageSize, TotalAverage)
    NUMBER_OF_RANDOM_FACES = 3;
    RandomFaces = ...
    randi([1 size(dataMatrix, 2)], NUMBER_OF_RANDOM_FACES, 1);
    [A_tilde, U_tilde, S_tilde, V_tilde] = RankReduce(U, S, V, rank);
    figure;
    for I = 1: NUMBER_OF_RANDOM_FACES
        FaceID  = RandomFaces(I);
        TheFace = dataMatrix(:, FaceID); 
        subplot(2, NUMBER_OF_RANDOM_FACES, I); 
        imshow(ArrayToGrayScale(TheFace, ImageSize));
        TheFaceReconstruct = A_tilde(:, FaceID);
        subplot(2, NUMBER_OF_RANDOM_FACES, NUMBER_OF_RANDOM_FACES + I); 
        imshow(ArrayToGrayScale(TheFaceReconstruct + TotalAverage, ImageSize));
    end
    sgtitle(["Reconstruct Random Faces using a rank of: ", num2str(rank)]);
end
