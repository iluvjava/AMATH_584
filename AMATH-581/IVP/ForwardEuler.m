function Fxnout = ForwardEuler(f, y0, dt)
% Timestepping using the forward Euler scheme. 
%   f : 
%       This is a function, possibly multi-variable, f(t, y)
%   y0: 
%       The initial condiction for Euler Time-stepping. 
%   dt: 
%       A vector of time series we want to do the time-stepping for.
%   Fxnout: 
%       The y values for each time series queried by dt, including the zero
%       for the initial condition, it should be monotone increasing 
%       sequences of time frames. 

    Fvalues = zeros(1, length(dt));
    Fvalues(1) = y0;
    for I = 2:length(dt)
        PreviousY = Fvalues(I - 1);
        Dt = dt(I) - dt(I - 1); 
        Fvalues(I) = PreviousY + Dt*f(dt(I), PreviousY);
    end
    Fxnout = Fvalues;
end