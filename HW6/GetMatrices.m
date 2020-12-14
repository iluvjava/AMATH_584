function [A, B, Images, Labels] = GetMatrices(toRead)
    % Read the A, B matrix from the data set as specified in the HW
    % assignment. 
    % toRead: 
    %   Binary variable, if it's 1, then it's the training set if it's 2
    %   then it's the validation set. 
    
    mode           = ["train", "t10k"];
    DataFile       = strcat("data\", mode(toRead), "-images.idx3-ubyte");
    Labels         = strcat("data\", mode(toRead), "-labels.idx1-ubyte");
    [Data, Labels] = MNISTRead(DataFile, Labels);
    Images         = Data;
    
    [m, n, q] = size(Data);
    A         = zeros(m*n, q); 
    for II = 1: size(Data, 3)
        A(:, II) = reshape(Data(:, :, II), m*n, 1);
    end
    
    B            = zeros(10, q);
    for II = 1: q
        if Labels(II) == 0
            B(10, II) = 1;
            continue;
        end
        B(Labels(II), II) = 1;
    end
    
end

