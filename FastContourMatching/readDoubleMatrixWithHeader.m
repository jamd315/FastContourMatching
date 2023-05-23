function [X] = readDoubleMatrixWithHeader(fname)

fp = fopen(fname, 'r');
if(fp < 0)
    fprintf('error opening file %s\n', fname);
    keyboard;
end
numrows = fscanf(fp, '%d', 1);
numcols = fscanf(fp, '%d', 1);
%fprintf('reading %d x %d matrix from %s\n', numrows, numcols, fname);
[data,count] = fscanf(fp, '%f ', numrows * numcols); 
if(count ~= (numrows*numcols))
    fprintf('error reading %s\n', fname);
    keyboard;
end
X = reshape(data, numcols, numrows)';

fclose(fp);
