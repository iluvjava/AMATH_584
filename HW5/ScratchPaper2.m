%% LOADING THE DATA MATRIX. 
clear all; close all; clc; 

% Get files
dirinfo = dir("..\HW2\yale-faces\yalefaces_cropped\CroppedYale\**");
dirinfo([dirinfo.isdir]) = [];

% Meta Setting, create matrix
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

%% Power Iteration 
clc; 
[U, S, V] = svd(ColumnDataMatrix, "econ"); % Ground truth
M = ColumnDataMatrix*ColumnDataMatrix.';

[EigenVectors, ~] = AutoPowerIterate(M);

figure; 
DominantFace = ArrayToGrayScale(abs(EigenVectors(:, end)), ImageSize);
imshow(DominantFace); title("power-itr-eigenface");
saveas(gcf, "Power-itr-faces", "png"); 

figure;
DominantFace = ArrayToGrayScale(abs(U(:, 1)), ImageSize);
imshow(DominantFace); title("Actual-Eigenface");
saveas(gcf, "actual-svd-eigenface", "png");

%% RANDOMIZED SVD 

RANKS       = 30: 400: 1630;
Plothandles = zeros(1, length(RANKS));
LegendNames = cell(1, length(RANKS));

figure; 
for II = 1: length(RANKS)                    % Variable to tweak.  
    r = RANKS(II);
    [UTilde, STilde, VTilde] = Rsvd(ColumnDataMatrix, r);     % NEEDS TO GET THE V^H too approximation too. 
    SingularValuesApprox     = diag(STilde); 
    h = semilogy(1:r, SingularValuesApprox, "x"); hold on;
    Plothandles(II) = h;
    LegendNames{II} = strcat("Approx with r: ", num2str(r));
end

SingularValuesActual = diag(S);
h = semilogy(1: r, SingularValuesActual(1: r), "bo");
LegendNames{II + 1} = "Actual";
legend(LegendNames);
title("Randomized Modes Decay Differenr Ranks");
saveas(gcf, "randomized-modes-decay", "png");


%% COMPARING THE MATRICES

% Trucate the Original U, S, V matrices according to the number of leading
% modes. 

UHat = U(:, 1: r); 
SHat = S(1: r, 1:r);
VHat = V(:, 1: r);

% Compare

ErrorU = norm(abs(UHat) - abs(UTilde));
ErrorS = norm(abs(SHat) - abs(STilde));
ErrorV = norm(abs(VHat) - abs(VTilde));

ErrorU = ErrorU./(size(UHat, 1)*size(UHat, 2));
ErrorS = ErrorS./(size(SHat, 1)*size(SHat, 2));
ErrorV = ErrorV./(size(VHat, 1)*size(VHat, 2));

disp("Error on recovered matrices: ")
disp(strcat("UMatrix Error: ", num2str(ErrorU)));
disp(strcat("SMatrix Error: ", num2str(ErrorS))); 
disp(strcat("VMatrix Error: ", num2str(ErrorV)));







 
