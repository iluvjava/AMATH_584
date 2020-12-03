% We are going to do some randomized Linear algebra for this shit. 
clear all; close all; clc; 

%% Get files
dirinfo = dir("..\HW2\yale-faces\yalefaces_cropped\CroppedYale\**");
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

%% Power Iteration 

M = ColumnDataMatrix*ColumnDataMatrix.';
%%
[EigenVectors, ~] = AutoPowerIterate(M);
%% 
DominantFace = ArrayToGrayScale(EigenVectors(:, end), ImageSize);
imshow(DominantFace); title("power-it-eigenface");
saveas(gcf, "Power-itr-faces", "png")

%% randomized SVD 
clc; 
[U, S, V] = svd(ColumnDataMatrix, "econ"); % Ground truth

RANKS        = 50: 400: 1700;
Plothandles  = zeros(1, length(RANKS));
LegendNames = cell(1, length(RANKS));

figure; 
for II = 1: length(RANKS)% variable to tweak.  
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