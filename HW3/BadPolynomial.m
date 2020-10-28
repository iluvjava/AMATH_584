function output = BadPolynomial(x)
    
    Coefficients = ... 
        [1, -18, 144, -627, 2016, -4032, 5376, -4608, 2304, -512]; 
    
    PowerAccumulate = ones(1, length(x));
    output = zeros(1, length(x)); 
    for C = Coefficients
        output = output + C*PowerAccumulate; 
        PowerAccumulate = PowerAccumulate.*x; 
    end
end

