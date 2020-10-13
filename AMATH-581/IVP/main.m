%% 
% Order of accuracy for the Euler, Heun's methods:
clear all; close all, clc;
hold on;
DeltaTs = 2.^(-2:-1:-8); 
Errors = zeros(1, length(DeltaTs));
Analytical = @(t) pi*exp(3*(cos(t) - 1))/sqrt(2);
f = @(t, y) -3*y*sin(t); y0 = pi/sqrt(2);
Ans = {};
for H = {@ForwardEuler, @HeunMethod}
    for I = 1:length(DeltaTs)
        TimeSeries = 0: DeltaTs(I): 5;
        YsAnalytical = Analytical(TimeSeries);
        YsNumerical = H{1}(f, y0, TimeSeries);
        Errors(I) = mean(abs(YsAnalytical - YsNumerical));
    end
    PolyFit = polyfit(log(DeltaTs), log(Errors), 1);
    plot(log(DeltaTs), log(Errors), "o-");
    title("Local Error Magnitude");
    ylabel("log(E)"); xlabel("log(\Delta t)");
    Ans{end + 1} = YsNumerical.';
    Ans{end + 1} = Errors;
    Ans{end + 1} = PolyFit(1);
end
legend("ForwardEuler", "HeunMethod", "location", "best");

% Stores the answers for the questions
A1 = Ans{1}; A2 = Ans{2}; A3 = Ans{3}; A4 = Ans{4};
A5 = Ans{5}; A6 = Ans{6};

%% 
% Van der Pol Oscillator 
function Fxnout = Oscillator(t, y)
    
end
