function Layers = VisualizeAllLayers(model)
    
    
    Layers = zeros(28, 28, 10);
    for II = 1: 10
       Layers(:, :, II) = reshape(model(II, :), size(Layers, [1, 2]));
    end
    
    DrawingBoard = zeros(28*2, 28*5);
    for II = 1: 10
        III = floor((II - 1)/5);
        JJJ = mod((II - 1), 5);
        DrawingBoard(III*28 + 1: (III + 1)*28, JJJ*28 + 1: (JJJ + 1)*28)...
            = Layers(:, :, II);
    end
    figure; pcolor(DrawingBoard);
    
end