function BestRank = EnergyAnalysis(s)
%   Function will use the SVD decomposition of the matrix to look for the 
%   number of singular values that gives 95% of the engery. 
    D = diag(s);
    TotalEnergy = cumsum(D)./sum(D);
    BestRank = min(find(TotalEnergy > 0.95));
    figure;
    semilogy(TotalEnergy, 'ok');
    ylabel("Energy");
    xlabel("Singular Value Rank");
    title("BestRank for Energy");
    xline(BestRank, '--');
    yline(TotalEnergy(BestRank));
end

