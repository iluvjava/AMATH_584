function Fxnout = HeunMethod(f, y0, ts)
% Timestepping using the forward Euler scheme. 
%   f : 
%       This is a function, possibly multi-variable, f(t, y)
%   y0: 
%       The initial condiction for Euler Time-stepping. 
%   ts: 
%       A vector of time series we want to do the time-stepping for.
%   Fxnout: 
%       The y values for each time series queried by ts, including the zero
%       for the initial condition   

    Ys = zeros(1, length(ts));
    Ys(1) = y0;
    for I = 2: length(ts)
        T = ts(I);
        Ypre = Ys(I - 1);
        F = f(ts(I - 1), Ypre);
        Ys(I) = Ypre + ...
            ((T - ts(I - 1))/2)* ... 
            (F + f(T, Ypre + (T - ts(I - 1))*F));
    end
    Fxnout = Ys;
end