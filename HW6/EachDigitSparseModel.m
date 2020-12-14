function [Model, Filters] = EachDigitSparseModel(datapool, Filters)
    % Theory: 
    %       Use the sparse filter for each digit and then use backslash to
    %       get the model. 
    % DataPool: 
    %       A datapool object.
    % Filters: 28 x 28 x 10
    %       10 Filters, for each digit, the filter should come from one of
    %       the sparse model using Lasso L1 Norm.
    %   
    
    if nargin == 1  % Compute the best filters.
        Filters = zeros(28, 28, 10);
        datapool.ApplySparseFilter(ones(28^2, 1));
        %datapool.Intercept = 0;
        for II = 1: 10
            disp(strcat("Computing Best Filter for digit: ", num2str(II)));
            Options            = optimset("display", "iter", "MaxFunEvals", 40);
            [LambdaOptimal, ~] = fminbnd(@(x) GetError(datapool, II, x), 0.00, 0.05, Options);
            [Filter, ~]        = datapool.LassoSingleDigitTrain(II, LambdaOptimal, 1);
            Filters(:, :, II)  = reshape(Filter, 28, 28);
        end
        %datapool.Intercept = 1;
    end
    
    Model = zeros(10, 28^2);
    for II = 1: 10
        datapool.ApplySparseFilter(Filters(:, :, II) ~= 0);
        A = datapool.DataMatrix;
        B = datapool.LabelMatrix; 
        B = B(II, :)';
        Model(II, :) = (A'\B)';
    end
    
    function Error = GetError(datapool, digit, lambda)
        [M, stats]     = datapool.LassoSingleDigitTrain(digit, lambda, 1);
        Predicted      = M'*datapool.Afull + stats.Intercept;
        Density        = sum(M ~= 0)/784;
        CorrectAnswers = datapool.Bfull(digit, :);
        Error          = norm(Predicted - CorrectAnswers)/(1 - Density);
    end 
end
