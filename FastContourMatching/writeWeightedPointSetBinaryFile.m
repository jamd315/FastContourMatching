function writeWeightedPointSetBinaryFile(fname, pointsets)
% Function to write an binary file containing weighted point sets in the following format:
% N - number of point sets in this file (int32)
% d - dimension of all the points in all the sets (int32)
% then for each of the N point sets:
%    n_i - the number of points in the ith set (int32)
%    x_1^1, x_1^2, ... , x_1^d, x_2^1, ..., x_2^d, ... , x_{n_i}^d -- the points in the ith set (floats)
%    w_1, ... , w_{n_i} (floats)

dim = size(pointsets(1).features,2);
N = length(pointsets);
for i=2:N
    if(size(pointsets(i).features,2)~=dim)
        error('all point sets must contain points of same dim');
    end
end

fp = fopen(fname, 'w');
fwrite(fp, N, 'int');
fwrite(fp, dim, 'int');
%fprintf('writing %d pointsets of dim %d to %s\n', N, dim, fname);
for i=1:N
    n = size(pointsets(i).features,1);
    fwrite(fp, n, 'int');
    fwrite(fp, pointsets(i).features', 'float');
    fwrite(fp, pointsets(i).weights, 'float');
end


fclose(fp);
