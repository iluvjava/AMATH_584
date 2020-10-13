function Fxout = ArrayToGrayScale(arr, size)
% This function plots an array into an gray scale image. 
% arr: 
%   A vector that represents all the datapoint of the image. 
% size:
%   2d array representing the size of the matrix. 
% tittle: 
%   String that is going to be the tile of the plot. 
   arr = arr - min(arr);
   Scale = 255/max(arr);
   Fxout = uint8(reshape(arr*Scale, size));
end