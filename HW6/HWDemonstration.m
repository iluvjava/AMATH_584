%% PRODUCING VARIOUS MODELS USING DIFFERENT SOLVERS
clear variables; clc; 

data        = TrainingDataPool(6000);
DataMatrix  = data.DataMatrix;
LabelMatrix = data.LabelMatrix; 


%% 
X1          = (DataMatrix'\LabelMatrix')';
X2          = (pinv(DataMatrix')*LabelMatrix')';
[X3, Beta3] = data.SingleLambdaLasso(0.008, 0.0001);  % Ridge Regression. 
[X4, Beta4] = data.SingleLambdaLasso(0.008, 1);       % Full L1
data.Intercept = 0;
XX3         = data.SingleLambdaLasso(0.008, 1);       % L1 Lasso no intercept. 
data.Intercept = 1; 


%% Model Assessment
X1Score      = data.GetModelScore(X1);
X2Score      = data.GetModelScore(X2);
X3Score      = data.GetModelScore(X3, Beta3); 
X4Score      = data.GetModelScore(X4, Beta4);
XX3Score     = data.GetModelScore(XX3);


%% VISUALIZING VARIOUS MODELS PRODUCED

VisualizeAllLayers(log(1./abs((X1 == 0) + X1))); colorbar; title("Backslash All Entries Log");
saveas(gcf, "backslash-log", "png");

VisualizeAllLayers(sign(abs(X1))); colorbar; title("Backslash Non zeros entry");
saveas(gcf, "backslash-nonzero", "png");

VisualizeAllLayers(log(1./abs((X2 == 0) + X2))); colorbar; title("Pinv All Entries Log");
saveas(gcf, "pinv-log", "png");

VisualizeAllLayers(sign(abs(X2))); colorbar; title("Pinv Non zeros entry");
saveas(gcf, "pinv-nonzero", "png");

VisualizeAllLayers(log(1./abs((X3 == 0) + X3))); colorbar; title("LassoRidge All Entries Log");
saveas(gcf, "lassoridge-log.png", "png");

VisualizeAllLayers(sign(abs(X3))); colorbar; title("LassoRidge  Non zeros entry");
saveas(gcf, "lassoridge-nonzero.png", "png");

VisualizeAllLayers(log(1./abs((X4 == 0) + X4))); colorbar; title("Lasso All Entries Log");
saveas(gcf, "lasso-log.png", "png");

VisualizeAllLayers(sign(abs(X4))); colorbar; title("LassoRidge Non zeros entry");
saveas(gcf, "lasso-nonzero.png", "png");

%% VARIABILITY IN LAMBDA
[B, Stats] = data.LassoSingleDigitTrainCV(1, logspace(-3, -2, 30), 1);

%%
lassoPlot(B, Stats, "PlotType", "CV");
legend("Show");
saveas(gcf, "digit-1-lasso-cv", "png")

%%
[B, Stats] = data.LassoSingleDigitTrainCV(5, logspace(-3, -2, 30), 1);

%%
lassoPlot(B, Stats, "PlotType", "CV");
legend("Show");
saveas(gcf, "digit-5-lasso-cv", "png");


%% SPARSE FILTER ALL DIGITS

Filter = VisualizeOverlayed(X4, 3); title("Important Pixels for All Digits")
saveas(gcf, "filter-alldigits", "png");
Filter  = Filter + zeros(28, 28, 10);
[X5, ~] = EachDigitSparseModel(data, Filter);
disp(strcat("Sparse model all digits scored: ", num2str(data.GetModelScore(X5)), "%"));
disp(strcat("Model Density: ", num2str(sum(sum(X5~=0))/7840)));
VisualizeAllLayers(X5 ~= 0); title("Sparse Model, All digits"); 
saveas(gcf, "sparse-all-digits", "png");

%% SPARSE EACH DIGITS DIFFERENT LAMBDA FOR EACH DIGITS
[X6, ~] = EachDigitSparseModel(data);
disp(strcat("Model Density: ", num2str(sum(sum(X6~=0))/7840)));
VisualizeAllLayers(X6 ~= 0); 
colorbar; title("Each Digit Different Lambda nonzero");
saveas(gcf, "each-digit-diff-lambda-log", "png");
VisualizeAllLayers(log(1./abs((X6 == 0) + X6))); 
colorbar; title("Each Digit Different Lambda log");
saveas(gcf, "each-digit-diff-lambda-nonzero", "png");
X6Score = data.GetModelScore(X6);
disp(strcat("Model Score: ", num2str(X6Score)));







