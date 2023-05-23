function [pcs, m, evals] = doPCA(X)
% takes the data matrix X (cols are points) and returns
% all principal components and mean

m = mean(X,2);

C = cov(X');  %cov function expects matrix where each row is an observation

[V,D] = eig(C);

% sort the eigenvalues/vectors according to magnitude of eigenvalues
evals = diag(D);

[sortedEvals, inds] = sort(evals);

pcs = zeros(size(V));

backInd = size(inds,1);

for i=1:size(inds,1)
    pcs(:,i) = V(:, inds(backInd));
    backInd = backInd - 1;
end

