classdef TrainingDataPool < handle
   
    properties
        Images;     % Image Tensor Full
        Labels;     % Image Label vector full
        Afull;      % Reshaped Data Matrix full 
        Bfull;      % Image lable matrix full
        
        % For training. 
        DataMatrix;
        LabelMatrix;
        ChosenSubset;
        
        A = nan;    % Flatten Tensor
        b = nan;    % Flatten Label vector
        
        TestMatrixB; % 1k label matrix. 
        TestMatrixA; % 1k Data matrix. 
        
        Intercept = 1;  % An extra option for Lasso Model, if this is one, 
                        % Then all lasso model are going to have intercept.
        
    end
    
    methods
        
        function this = TrainingDataPool(N)
            % Constructor. 
            [this.Afull, this.Bfull, this.Images, this.Labels] = ...
            GetMatrices(1);
            if N > length(this.Labels)/2
               error("No! Reserve 50% of them for cross validations.");
            end
            this.Scramble(N);
            [this.TestMatrixA, this.TestMatrixB, ~, ~] = ...
            GetMatrices(2);
        
        end
        
        function void = Scramble(this, N)
            % Scramble all the data and labels for training, N of them. 
            RandomIdx         = randperm(length(this.Labels));  % No repreatition 
            this.ChosenSubset = RandomIdx(1: N);
            this.DataMatrix   = this.Afull(:, this.ChosenSubset);
            this.LabelMatrix  = this.Bfull(:, this.ChosenSubset);
            void              = nan;
        end
        
        % Swap to Binary Label with +1, -1 for the label matrix. 
        function void = SwapLabelBinaryMode(this)
            B                = this.Bfull(:, this.ChosenSubset); 
            B                = -(B == 0) + B;
            this.LabelMatrix = B;
            void             = 0;
        end
        
        % Swap back to 0, 1 label. 
        function void = SwapLabelUnaryMode(this)
            this.LabelMatrix  = this.Bfull(:, this.ChosenSubset);
           void               = 0; 
        end
        
        function [A, b] = FlattenTensor(this)
            % Get a matrix A, such that, solving Ax = b gives x that is a
            % flattened linear model of size 784 by 10. This is for using
            % lasso that does regression all digits at the same time. 
            if ~isnan(this.A)
               A = this.A; 
               b = this.b;
               return
            end
            B = this.LabelMatrix;
            A = kron(eye(10), this.DataMatrix');
            b = reshape(this.LabelMatrix', size(B, 1)*size(B, 2), 1);
            
        end
        
        % Recover the 7840 by 1 matrix to a 10 by 784 model. 
        function X = VectorRecover(this, Vector)
            % Recover vectorized linear model from the flattened vector. 
            X = reshape(Vector, size(this.Images, 1)*size(this.Images, 2), 10);
            X = X';
        end
        
        
        % Given a model 10 by 764 by, and get the prediction made by this
        % model when we multiply it on the right handside by the
        % DataMatrix. 
        function [Predicted, Error] = Type1ModelPredictDiscrete(this, model)
            Intercept = zeros(10, 1);
            [Predicted, Error] = Type1ModelPredictDiscreteIntercept(this, model, Intercept);
        end
        
        % Predict the 10 by 784 model with an intercept vector. 
        function [Predicted, Error] = Type1ModelPredictDiscreteIntercept(this, model, intercept)
            
            % Predict on ALL DATA with a given Type I model.
            Predicted = model*this.TestMatrixA + intercept;
            NewMatrix = zeros(size(Predicted));
            
            for ColIndex = 1: size(NewMatrix, 2)
               [~, Idx]                 = max(Predicted(:, ColIndex));
               NewMatrix(Idx, ColIndex) = 1;
            end
            Predicted = NewMatrix;
            Error     = sum(sum(abs(Predicted - this.TestMatrixB)));  
        end
        
        % Get the percentable of correctly predictor lable on the MNIST
        % test set. the model given 10 by 784 matrix. 
        function Score = GetModelScore(this, model, intercept)
                if nargin == 2
                   intercept = zeros(10, 1);
                end
                Predicted  = this.Type1ModelPredictDiscreteIntercept(model, intercept);
                ModelScore = sum(sum(abs(Predicted - this.TestMatrixB)));
                Score      = 1 - ModelScore./(2*size(this.TestMatrixB, 2));
        end
        
        function Errors = ModelsErrorsTypeI(this, models)
            % Given a series of linear model, return all the errors on prediction. 
            % Models: 
            %       10 by 784 by N

            Errors = zeros(1, size(models, 3));
            for II = 1: size(models, 3)
                [~, Error] = this.Type1ModelPredictDiscrete(models(:, :, II));
                Errors(II) = Error;
            end
        end
        
        function ModelSeries = MultiLambdaLasso(this, Lambdas)
            % Given multiple lambdas, returns a series of 10 by 764
            % matrices for each of the given lambdas. 
            [m, n] = size(this.Images(:, :, 1));
            p      = length(Lambdas);
            A      = this.DataMatrix; 
            B      = this.LabelMatrix;
            Models = zeros(10, m*n, p);
            
            for II = 1: 10  % 10 rows. 
                % [Outputs, ~] = lasso(A', B(II, :)', "lambda", Lambdas);
                [Outputs, ~] = this.LassoSingleDigitTrain(II, Lambdas, 1);
                
                for JJ = 1: length(Lambdas)
                   Models(II, :, JJ) = Outputs(:, JJ);
                end
            end
            ModelSeries = Models;
        end
        
        function [Model, Intercept] = SingleLambdaLasso(this, lambda, alpha)
            % Given a value for lambda, get a 10 by 784 model for all the
            % all the digits with the same lambda. 
            if nargin == 2
               alpha = 1;
            end
            [m, n] = size(this.Images(:, :, 1));
            A      = this.DataMatrix; 
            B      = this.LabelMatrix;
            Model  = zeros(10, m*n);
            Intercept  = zeros(10, 1);
            for II = 1: 10                % 10 rows. 
                [Outputs, Info]  = this.LassoSingleDigitTrain(II, lambda, alpha);
                Model(II, :) = Outputs;
                if this.Intercept == 1
                    Intercept(II) = Info.Intercept;
                end
            end
        end
        
        % Train and get model for a single digit.
        % This is a subroutine that got repeatedly use hence it's been
        % factored out here. 
        function [SingleDigitModel, Stats] = LassoSingleDigitTrain(this, idx, lambdas, alpha)
            A                = this.DataMatrix; 
            B                = this.LabelMatrix;
            [Outputs, Info]  = lasso(A', B(idx, :)', ... 
                "lambda", lambdas, "alpha", alpha, "Intercept", boolean(this.Intercept));
            SingleDigitModel = Outputs;
            Stats            = Info;
        end
        
        % Train a single digit model with given regularizer. 
        function [SingleDigitModel, Stats] = LassoSingleDigitTrainCV(this, idx, lambdas, alpha)
            A                = this.DataMatrix; 
            B                = this.LabelMatrix;
            [Outputs, Info]  = lasso(A', B(idx, :)', "lambda", lambdas, "alpha", alpha, "CV", 20);
            SingleDigitModel = Outputs;
            Stats            = Info;
        end
        
        % Perform Robust fit on all digits. 
        function [Model, Stats] = RobustFit(this)
            [m, n] = size(this.Images(:, :, 1));
            A      = this.DataMatrix;
            B      = this.LabelMatrix;
            Model  = zeros(10, m*n);
            Stats  = cell(1, 10);
            for II = 1: 10  % 10 rows. 
                [Outputs, Info] = robustfit(A', B(II, :)', [], [], "off");
                Model(II, :) = Outputs;
                Stats{II}    = Info;
                disp("Robust fit...")
            end
        end
        
        % Filter out things with the sparse model
        % Filter is a binary matrix. 
        % Return a sparse data matrix and lable matrix, replace the field 
        % with the sparse data matrix, without changing the size of it.
        % (So it's full of zeroes in it. )
        % Call it without any argument and it will restore dense data
        % matrix.
        function [SparseDataMatrix] = ApplySparseFilter(this, Filter)
            if isequal(size(Filter), [28, 28])
                Filter = reshape(Filter, 28^2, 1);  % Shape to column vector. 
            end
            AChosen           = this.Afull(:, this.ChosenSubset); 
            this.DataMatrix   = Filter.*AChosen; 
            SparseDataMatrix  = this.DataMatrix;
        end
    end
end


