clc; clear variables; 
% Load the Training set. 

N    = 60000;  % When N > 784, over determined, when N < 784, underdetermined. 
data = TrainingDataPool(N);
A    = data.DataMatrix;
B    = data.LabelMatrix;

%% DIRECT SOLVE FOR A LINEAR MODEL
% Linear Mapping: R^784 --> R^10, then 10 by 784, then XA = B is the model
% description.. 
X1   = (A'\B')';
X2   = (pinv(A')*B')';


%% MULTI LASSO TRAIN
figure;
N = 1000;
for II = 1: 10
    data.Scramble(N);  % Batch size
    Lambdas     = linspace(0.001, 0.01, 10);
    ModelSeries = data.MultiLambdaLasso(Lambdas);
    Errors      = data.ModelsErrorsTypeI(ModelSeries);
    Errors      = Errors./(10000*2);  % Across 1k test data set. 
    plot(Lambdas, Errors); hold on;
    disp(strcat("Training Lasso: ", num2str(II)));
end
% Lambda Optimal: 0.004;


%% LASSO MODEL TRAIN ALL
data.Scramble(6000);
data.ApplySparseFilter(ones(28, 28));  % CLEAR THE FILTERS!
%%
[X3, StatsX3] = data.SingleLambdaLasso(0.004);
%%
[X4, StatsX4] = data.SingleLambdaLasso(0.004, 0.8);
%%
[X5, StatsX5] = data.SingleLambdaLasso(0.004, 0.5);
%%
% [X6, StatsX6] = data.RobustFit();
[X6, StatsX6] = data.SingleLambdaLasso(0.004, 0.01);
%% 
[X7, StatsX7] = data.SingleLambdaLasso(0.01);

%% SCORING BOARD
close all;
Score = data.GetModelScore(X1); 
VisualizeOverlayed(X1); title("X1");
disp(strcat("BackSlack Model score: ", num2str(Score)));

Score = data.GetModelScore(X2);
VisualizeOverlayed(X2); title("X2"); 
disp(strcat("Puesdo Inverse Model Score: ", num2str(Score)));

Score = data.GetModelScore(X3);
VisualizeOverlayed(X3); title("X3");
disp(strcat("Lasso Single score: ", num2str(Score)));

Score = data.GetModelScore(X4);
VisualizeOverlayed(X4); title("X4");
disp(strcat("Lasso With Alpha 0.8: ", num2str(Score)));

Score = data.GetModelScore(X5);
VisualizeOverlayed(X5); title("X5");
disp(strcat("Lasso With Alpha 0.5: ", num2str(Score)));

Score = data.GetModelScore(X6);
VisualizeOverlayed(X6); title("X6");
disp(strcat("Ridge Regression: ", num2str(Score)));

Score = data.GetModelScore(X7);
VisualizeOverlayed(X7); title("X7");
disp(strcat("Extreme Lasso Regression: ", num2str(Score)));
%% SPARSE MODEL VISUALIZATION!!!

VisualizeAllLayers(X7); title("X7")
%%
VisualizeAllLayers(X4); title("X4")
%%
VisualizeAllLayers(X3); title("X3")


%% SPARSE MODEL TRAINING!!!
clc;
data.Scramble(6000);
Filter = VisualizeOverlayed(X3, 5);
disp(strcat("Filter Density: ", num2str(sum(sum(Filter))./(28^2))));
data.ApplySparseFilter(Filter);
A        = data.DataMatrix; 
B        = data.LabelMatrix;
X1Sparse = (A'\B')';

VisualizeAllLayers(X1Sparse);
Score = GetModelScore(data, X1Sparse); 
disp(strcat("BackSlash Sparse Model score: ", num2str(Score)));

%% FULL SPARSE MODEL FOR EACH DIGITS!
clc; 
data.Scramble(6000);
[X8, X8Filters] = EachDigitSparseModel(data);
disp(strcat("Filter Density: ", num2str(sum(sum(X8~=0))./(10*28^2))));
VisualizeAllLayers(X8);

Score = data.GetModelScore(X8); 
disp(strcat("BackSlash Sparse Model each digit: ", num2str(Score)));

%% HELPER FUNCTIONS




