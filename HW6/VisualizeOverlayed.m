function Filter = VisualizeOverlayed(model, threshold)  % Visualize the model. 
    if nargin == 1
        threshold = 5;
    end

    model    = model ~= 0;  % Important! 
    SummedUp = zeros(1, size(model, 2)); 
    for II = 1: 10
       SummedUp = SummedUp + model(II, :);
    end
    SummedUp = reshape(SummedUp, 28, 28);
    Filter   = SummedUp >= threshold;   % Threshold 
    figure;
    pcolor(Filter); colorbar;
end