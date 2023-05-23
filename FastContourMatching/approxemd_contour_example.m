






basefname = '/data/vip/vh/emd/saved/entirePoserData/poserdb'; %% fill in path and filenames here
ptsetsfname = '/csail/vision-trevor6/kgrauman/tempptsets.bin';
costsoutname = '/csail/vision-trevor6/kgrauman/tempcosts.txt';


NUM_FILES = 2; % up to 20  
NUM_FILES_FOR_PCA = 2; % up to 20
numperfilepca = 350; % up to 7000 per file (140,000 examples total)
numperfile = 350;
DISC = 0;  % discretization factor - if 0, means don't change data when computing histograms
reducedDim = 8; % what to project to with PCA





% collect some histograms to run PCA on
X = [];
for i=1:NUM_FILES_FOR_PCA
    
    inname = [basefname int2str(i) '.mat'];
    fprintf('reading %s\n', inname);
    load(inname, 'poserdata');

    inds = randperm(length(poserdata));
    inds = inds(1:numperfilepca);
    
    for j=1:length(inds)
        X = [X, poserdata{inds(j)}.hist'];
    end
    clear poserdata
end



% compute PCA bases
[d,n] = size(X);
fprintf('%d pts of dimension %d\n', n, d);
[pcs, m, evals] = doPCA(X);

clear X



% go back and get some data to test with, projecting with bases just computed
outind = 1;
for i=1:NUM_FILES
    
    inname = [basefname int2str(i) '.mat'];
    load(inname, 'poserdata');

    inds = randperm(length(poserdata));
    inds = inds(1:numperfile);
    
    for j=1:length(inds)
        ptsets(outind).features = getProjections(poserdata{inds(j)}.hist', m, reducedDim, pcs)';
        n = size(ptsets(outind).features,1);
        if(size(ptsets(outind).features,2)~=reducedDim)
            fprintf('size error in proj\n');
            keyboard;
        end
        ptsets(outind).weights = ones(n,1);
        
        pts{outind} = poserdata{inds(j)}.respts;  % keep track of the contour points that are associated with these examples

        outind = outind + 1;
    end
    clear poserdata
end


% write a file that the c code can read in
writeWeightedPointSetBinaryFile(ptsetsfname, ptsets);



% make the following call from the command line
callname = ['./approxemd ' ptsetsfname ' ' costsoutname ' ' int2str(DISC)];
fprintf('call\n%s\nfrom the command line,', callname);
fprintf('then type dbcont when done\n');
keyboard;



% read in the resulting distance matrix (n x n for the n pointsets)
C = readDoubleMatrixWithHeader(costsoutname);



% view some of the nearest neighbors according to this distance matrix
k = 10;
for i=1:size(C,1)
    [nndist,nnind] = sort(C(i,:));
    for j=1:k
        subplot(1,k,j);
        contourpts = pts{nnind(j)};
        plot(contourpts(:,1),contourpts(:,2), 'r.'), title(num2str(nndist(j)));
        axis ij;
        axis equal;
    end
    fprintf('showing nn for example %d\n', i);
    pause;
   
end
