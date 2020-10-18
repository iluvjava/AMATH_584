function BestRank = EnergyAnalysis(s, threshold)
%   Function will use the SVD decomposition of the matrix to look for the 
%   number of singular values that gives 95% of the engery. 
    switch nargin
        case 2
            % pass
        case 1
            threshold = 0.95
        otherwise
            error("must be 1, or 2 parameters. ")
    end

    D = diag(s);
    TotalEnergy = cumsum(D)./sum(D);
    BestRank = min(find(TotalEnergy > threshold));
    figure;
    semilogy(TotalEnergy, 'ok');
    ylabel("Energy");
    xlabel("Singular Value Rank");
    title("BestRank for Energy");
    xline(BestRank, '--');
    yline(TotalEnergy(BestRank));
end

