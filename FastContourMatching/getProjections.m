function [projs] = getProjections(dataMat, m, k, pcs)
% takes matrix of point lists (concattenated views, resampled), mean (m) and principal
% components, and returns matrix of projections (each col of proj is the corresponding
% projection for each col of dataMat

numpts = size(dataMat, 2);

pcs = pcs(:,1:k);

projs = zeros(k, numpts);

repMean = repmat(m, 1, numpts);

diffs = dataMat - repMean;

for i=1:numpts
    projs(:,i) = pcs' * diffs(:,i);
end