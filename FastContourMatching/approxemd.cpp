#define FINESTSIDE 0.5
#define SIDELENGTH_FACTOR 2
#define DO_TRANSLATIONS 1
#define NORM_BY_MIN_CARD 1
#define MAX_INDEX_AT_MAX_LEVEL_PER_DIM 99999999
#define MAX_INPUT_FNAME_LENGTH 500

#include <sys/types.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>
#include <algorithm>
#include <assert.h>
#include <float.h>
#include <math.h>
#include <map>

using namespace std;

typedef struct
{
  int32_t* index;  //array of ints gives single index
  float value;
  int32_t dim;
}indexValuePair_t;

typedef std::vector<indexValuePair_t*> sparseVec_t;  
typedef std::vector<sparseVec_t> multireshist_t;  // indexed by level

typedef struct
{
  float** features;
  float* weights;
  int numFeatures;
  int featureDim;
} pointSet_t;



bool entryIndexLessThan(indexValuePair_t* i1, indexValuePair_t* i2)
{
    int32_t dim = i1->dim;
    for(int32_t i=0; i < dim; i++){
      if(i2->index[i] < i1->index[i])
	return 0;  // i2 > i1
      else if(i1->index[i] < i2->index[i])
	return 1;
      // otherwise, equal index at this dimension, so consider next dimension
    }
    // if equal completely, return 0
    return 0; 
}


bool entryIndexEqual(indexValuePair_t* i1, indexValuePair_t* i2)
{
    int32_t dim = i1->dim;
    for(int32_t i=0; i < dim; i++){
      if(i2->index[i] != i1->index[i])
	return 0; 
    }
    return 1; 
}


double L1(sparseVec_t* v1, sparseVec_t* v2)
{
  int32_t indexDim = (*v1)[0]->dim;

  int32_t n1 = v1->size();
  int32_t n2 = v2->size();
  
  indexValuePair_t entry1,entry2;
  entry1 = (*(*v1)[0]);
  entry2 = (*(*v2)[0]);

  indexValuePair_t MAX_ENTRY;
  MAX_ENTRY.dim = indexDim;
  MAX_ENTRY.value = 0.0;
  MAX_ENTRY.index = new int[indexDim];
  for(int i=0; i < indexDim; i++){
    MAX_ENTRY.index[i] = MAX_INDEX_AT_MAX_LEVEL_PER_DIM;
  }

  int32_t ind1 = 0;
  int32_t ind2 = 0;
  int32_t gtrind = -1;

  double s = 0.0;
 
  while( (ind1 < n1) | (ind2 < n2) ){
    
    if(entryIndexEqual(&entry1,&entry2)){

      // add L1 distance between these entries, where bin indices are the same
      s += fabs(entry1.value - entry2.value);

      ind1++;

      if(ind1 >= n1)
	entry1 = MAX_ENTRY;
      else{
	entry1 = (*(*v1)[ind1]);
      }
      ind2++;

      if(ind2 >= n2)
	entry2 = MAX_ENTRY;
      else{
	entry2 = (*(*v2)[ind2]);
      }
    }
    else{
      
      if(entryIndexEqual(&entry1, &MAX_ENTRY))
	gtrind = 0;
      else if(entryIndexEqual(&entry2, &MAX_ENTRY))
	gtrind = 1;
      else if(entryIndexLessThan(&entry1, &entry2))
	gtrind = 1;
      else
	gtrind = 0;
      if(gtrind == 1){ // if index1 < index 2

	s += entry1.value;
	
	ind1++;

	if(ind1 >= n1)
	  entry1 = MAX_ENTRY;
	else{
	  entry1 = (*(*v1)[ind1]);
	}
      }
      else{ // index2 < index1

	s += entry2.value;

	ind2++;
	
	if(ind2 >= n2)
	  entry2 = MAX_ENTRY;
	else{
	  entry2 = (*(*v2)[ind2]);
	}
      }
    }
  }
  delete[] MAX_ENTRY.index;
  return s;
}


double getL1CostValue(multireshist_t* h1, multireshist_t* h2, float sidelengthFactor, float finestSidelength)
{
  assert(h1->size() == h2->size()); // hists must have same number of levels
  
  int32_t dim = (*h1)[0][0]->dim;
  assert(dim == (*h2)[0][0]->dim);

  int numLevels = h1->size();
  double dist, weight;
  int i;

  // get the distances for each level 
  double cost = 0.0;
  for(i=0; i < numLevels; i++){
    weight = finestSidelength * pow(sidelengthFactor,i);
    dist = L1(&(*h1)[i],&(*h2)[i]);
    cost += weight * dist;
  }

  return cost;  
}





void deletePointSets(pointSet_t* ps, int32_t numPointsets)
{
  for(int32_t i=0; i < numPointsets; i++){
    for(int32_t j=0; j < ps[i].numFeatures; j++){
      delete [] ps[i].features[j];
    }
    delete[] ps[i].features;
    delete[] ps[i].weights;
  }
  delete[] ps;
}


// get an n x n distance matrix for the n example hists
void getL1CostMatrix_oneset(double*** D, vector<multireshist_t>* hists, float sidefactor, float finestSidelength)
{
  int32_t i,j,n;
  n = hists->size();

  // allocate matrix for distance values
  fprintf(stderr, "Allocating %d x %d distance matrix...", n, n);
  (*D) = new double*[n];
  for(i=0; i < n; i++){
    (*D)[i] = new double[n];      
  }
  fprintf(stderr, "done.\n");
  double d;

  FILE* fp;

  // compute the cost values between all pairs of examples
  fprintf(stderr, "Computing cost values...\n");
  for(i = 0; i < n; i++){
    clock_t timestart = clock();
    for(j = i; j < n; j++){
      if(j != i){
	d = getL1CostValue(&(*hists)[i], &(*hists)[j], sidefactor, finestSidelength);
	assert(d > 0);
      }
      else{
	d = 0.0; // against self, distance = 0.
      }
      (*D)[i][j] = d;
      (*D)[j][i] = d;
    }
    clock_t timeend = clock();
    double rowTime = (double) (timeend - timestart) / CLOCKS_PER_SEC;
    double meanPerTime = rowTime / (double)(n-i);
    fprintf(stderr, "Computed row %d of %d, took %lf seconds: on average %lf seconds per example\n", i, n, rowTime, meanPerTime);
  }  

}





void deleteSparseVec(sparseVec_t* v){
  int32_t n = v->size();
  for(int32_t i=0; i < n; i++){
    delete[] (*v)[i]->index;
    delete (*v)[i];
  }
}


void sortAndTallyHistogramVector(sparseVec_t* inputvec, vector<multireshist_t>* outputhists, int outputindex, int level)
{
  int32_t i,j,k,dim;
  dim = (*inputvec)[0]->dim;

  std::sort(inputvec->begin(), inputvec->end(), entryIndexLessThan);

  if(&(*outputhists)[outputindex][level] != NULL)
    deleteSparseVec(&(*outputhists)[outputindex][level]);   

  (*outputhists)[outputindex][level].clear();

  indexValuePair_t* nextEntry = new indexValuePair_t;
  nextEntry->dim = dim;
  nextEntry->value = 0.0;
  nextEntry->index = new int32_t[dim];
  for(k=0; k < dim; k++){
    nextEntry->index[k] = (*inputvec)[0]->index[k];
  }
  j = 0;

  while(1){
   
    while(entryIndexEqual(nextEntry, (*inputvec)[j])){ // while repeating an index, sum the weights/value attached to that index
      nextEntry->value += (*inputvec)[j]->value;
      j++;
      if(j > (inputvec->size()-1) )
	break;
    }

    (*outputhists)[outputindex][level].push_back(nextEntry);
  
    if(j > (inputvec->size()-1) )
      break;

    nextEntry = new indexValuePair_t;
    nextEntry->dim = dim;
    nextEntry->value = 0.0;
    nextEntry->index = new int32_t[dim];
    for(k=0; k < dim; k++){
      nextEntry->index[k] = (*inputvec)[j]->index[k];
    }
  }
}


void writeDoubleMatrix(double** mat, int numrows, int numcols, char* outname)
{
  FILE* fp = fopen(outname, "w");
  if(fp == NULL){
    fprintf(stderr, "writeDoubleMatrix: Null file pointer to %s\n", outname);
    exit(1);
  }
  fprintf(stderr, "Writing %d x %d matrix of doubles to %s\n", numrows, numcols, outname);

  // write header
  fprintf(fp, "%d %d\n", numrows, numcols);
    
  for(int i=0; i < numrows; i++){
    for(int j=0; j < numcols; j++){
      fprintf(fp, "%lf ", mat[i][j]);
    }
    fprintf(fp, "\n");
  }

  fclose(fp);
}


void getMultiresHistograms(pointSet_t* pointsets, int32_t numPointsets, int32_t discretizeOrderIncrease, float minRawFeatureVal,
			   float** translations, int numLevels, int dim, float finestSidelength, float sidefactor, vector<multireshist_t>* hists)
{
  int32_t i,j,k,d,p;
    
  // check that all sets have same dim and get the number of points in the smallest set
  for(i=0; i < numPointsets; i++){
    if(pointsets[i].featureDim != dim){
      fprintf(stderr, "getMultiresHistograms error: all point sets must have points of the same dimension (%d here)\n", dim);
      exit(1);
    }
  }
  


  // discretize the point set features to integers
  float factor = pow(10,discretizeOrderIncrease);
  for(i=0; i < numPointsets; i++){
    assert(pointsets[i].numFeatures > 0);
    for(j=0; j < pointsets[i].numFeatures; j++){
      for(k=0; k < dim; k++){
	  pointsets[i].features[j][k] = round( (pointsets[i].features[j][k] - minRawFeatureVal)*factor);
      }
    }
  }
  



  // Form the histograms using the given translations and levels
  hists->resize(numPointsets);

  sparseVec_t tempVec;  
  int32_t index;
  
  for(i=0; i < numPointsets; i++){

    (*hists)[i].resize(numLevels);

    float gridsize = finestSidelength;
    
    // form a histogram for this point set at each level
    for(k=0; k < numLevels; k++){
   
      tempVec.clear();
   
      if(!(*hists)[i][k].empty())  // hists[i][k] is kth level histogram for point set i, is a vector of ind-val pairs
	(*hists)[i][k].clear();

      for(j=0; j < pointsets[i].numFeatures; j++){

	//allocate next nonzero entry for the embedding
	indexValuePair_t* nextNonzeroEntry = new indexValuePair_t;
	nextNonzeroEntry->dim = dim;
	nextNonzeroEntry->index = new int32_t[dim];
 
	for(d=0; d < dim; d++){
	  index = (int) ((pointsets[i].features[j][d] - translations[k][d]) / gridsize);
	  nextNonzeroEntry->index[d] = index;
	}
	nextNonzeroEntry->value = pointsets[i].weights[j];
	tempVec.push_back(nextNonzeroEntry); // push on a pointer to this new struct / nonzero entry
      }//end for j=1:numfeatures

      assert(tempVec.size() == pointsets[i].numFeatures);

      // sort the vector entries by index and sum the values of those entries with the same index.
      sortAndTallyHistogramVector(&tempVec, hists, i, k);
    
      deleteSparseVec(&tempVec);  

      gridsize = gridsize * sidefactor;

    }//end for k=1:numlevels
   }//end for i=1:numpointsets  

}




void getMinMaxFeatureValue(pointSet_t* ptsets, int32_t numptsets, float* minValue, float* maxValue)
{
  float minv = FLT_MAX;
  float maxv = -1 * FLT_MAX;

  for(int32_t i=0; i < numptsets; i++){
    for(int32_t j= 0; j < ptsets[i].numFeatures; j++){
      for(int32_t k=0; k < ptsets[i].featureDim; k++){
	if(ptsets[i].features[j][k] < minv)
	  minv = ptsets[i].features[j][k];
	if(ptsets[i].features[j][k] > maxv)
	  maxv = ptsets[i].features[j][k];
      }
    }
  }
  (*maxValue) = maxv;
  (*minValue) = minv;
}






void prepareHistogramSettings(float minRawFeatureVal, float maxRawFeatureVal, int dim, float finestSidelength, float sidefactor,
			      int32_t discretizeOrderIncrease, int* numLevels, float*** translations, int doTrans, double* diameter){
  int i,j;
  
  // calculate what the discretized max will be (min will be 1.0)
  float factor = pow(10,discretizeOrderIncrease);
  float maxDiscreteValue = round( (maxRawFeatureVal - minRawFeatureVal)*factor) + 1;
  
  // get the diameter, and num levels
  (*diameter) = maxDiscreteValue;
  (*numLevels)  = (int) round(log((*diameter) / finestSidelength) / log(sidefactor)) + 2;
  
  fprintf(stderr, "discretize factor = %f, maxDiscreteValue = %f, diameter = %lf, numlevels = %d\n",
	  factor, maxDiscreteValue, (*diameter), (*numLevels));
  assert((*numLevels) > 0);

  // return the random translations to use and the number of levels- numlevels x dim matrix 
  (*translations) = new float*[(*numLevels)];
  for(i=0; i < (*numLevels); i++)
    (*translations)[i] = new float[dim];
  
  
  srand(time(NULL));
  
  if(doTrans){
    // same shifts at all levels
    for(j=0; j < dim; j++){
      float randshift = ( (float) rand() / RAND_MAX) * (*diameter);
      for(i=0; i < (*numLevels); i++){
	(*translations)[i][j] = randshift;
      }
    }
  }
  else{
    // no shifts
    for(i=0; i < (*numLevels); i++){
      for(j=0; j < dim; j++){
	(*translations)[i][j] = 0.0;
      }
    }
  }
}
 






// Function to read a binary-format file containing weighted point sets in the following format:
// N - number of point sets in this file (int32)
// d - dimension of all the points in all the sets (int32)
// then for each of the N point sets:
//    n_i - the number of points in the ith set (int32)
//    x_1^1, x_1^2, ... , x_1^d, x_2^1, ..., x_2^d, ... , x_{n_i}^d -- the points in the ith set (floats)
//    w_1, ... , w_{n_i} (floats)
 void readWeightedPointSetsBinaryFile(char* fname, int32_t* numPointsets, pointSet_t** pointSets){
  FILE* fp = fopen(fname, "r");
  if(fp == NULL){
    fprintf(stderr, "Null file pointer to %s\n", fname);
    exit(1);
  }
  int32_t dim;
  int32_t n;
  int i,j,k;

  fread(numPointsets, sizeof(int32_t), 1, fp);
  fread(&dim, sizeof(int32_t), 1, fp);

  (*pointSets) = new pointSet_t[(*numPointsets)];
  
  for(i=0; i < (*numPointsets); i++){
    
    fread(&n, sizeof(int32_t), 1, fp);
 
    (*pointSets)[i].features = new float*[n];
    (*pointSets)[i].weights = new float[n];
    (*pointSets)[i].numFeatures = n;
    (*pointSets)[i].featureDim = dim;

     // read the n points in this set
    for(j=0; j < n; j++){
      (*pointSets)[i].features[j] = new float[dim];
      for(k=0; k < dim; k++){
	fread(&(*pointSets)[i].features[j][k], sizeof(float), 1, fp);
	assert(!isnan((*pointSets)[i].features[j][k]));
      }
    }
    // read the n weights for this set
    for(j=0; j < n; j++){
      fread(&(*pointSets)[i].weights[j], sizeof(float), 1, fp);
      assert(!isnan((*pointSets)[i].weights[j]));
    }
  }
  fclose(fp);
}










main(int argc, char** argv)
{
  
  if(argc != 4){
    fprintf(stderr, "usage: ./approxemd <1. ptsetsInname> <2. costsOutname> <3. discretizeOrder>\n");
    exit(1);
  }
  
  // read in the input parameters
  char inputFname[MAX_INPUT_FNAME_LENGTH];
  strcpy(inputFname, argv[1]);
  char costsOutname[MAX_INPUT_FNAME_LENGTH];
  strcpy(costsOutname, argv[2]);
  int discretizeOrderIncrease = atoi(argv[3]);


  int32_t dim;
  int32_t numPointsets;
  pointSet_t* pointsets;
  float minRawFeatureValue, maxRawFeatureValue;
  int numLevels;
  float** translations;
  double diameter;
  vector<multireshist_t> pyramids;
  clock_t timestart, timeend;
  double t;
  double** D;


  
  // read input point sets
  fprintf(stderr, "Reading point sets from file %s...\n", inputFname);
  readWeightedPointSetsBinaryFile(inputFname, &numPointsets, &pointsets); 
  fprintf(stderr, "Read %d point sets from %s.\n", numPointsets, inputFname);
  
    

  // determine min and max feature values in order to discretize, set diameter
  fprintf(stderr, "Getting min and max feature values...\n");    
  getMinMaxFeatureValue(pointsets, numPointsets, &minRawFeatureValue, &maxRawFeatureValue);  
  dim = pointsets[0].featureDim;
  fprintf(stderr, "Raw feature values: min %f, max %f\n", minRawFeatureValue, maxRawFeatureValue);
  fprintf(stderr, "Dimension of feature vectors is %d\n", dim);
  

  // form the histograms
  fprintf(stderr, "Preparing histogram settings (using translations=%d)...\n", DO_TRANSLATIONS);
  prepareHistogramSettings(minRawFeatureValue, maxRawFeatureValue, dim, FINESTSIDE, SIDELENGTH_FACTOR,
			   discretizeOrderIncrease, &numLevels, &translations, DO_TRANSLATIONS, &diameter);
  fprintf(stderr, "Setting numLevels = %d, diameter = %lf.\n", numLevels, diameter);
  



  // get the histogram pyramid for each pointset
  fprintf(stderr, "Getting multires histograms from %d pointsets - discretize order %d, finest side len %f\n",
	  numPointsets, discretizeOrderIncrease, FINESTSIDE);
  timestart = clock();
  getMultiresHistograms(pointsets, numPointsets, discretizeOrderIncrease, minRawFeatureValue,
			translations, numLevels, dim, FINESTSIDE, SIDELENGTH_FACTOR, &pyramids);
  timeend = clock();
  t = (double) (timeend-timestart) / CLOCKS_PER_SEC;
  fprintf(stderr, "Time to compute pyramids for %d pointsets is %lf seconds\n", numPointsets, t);
  
         


  // compute the cost values
  fprintf(stderr, "Computing the distance matrix...\n");
  getL1CostMatrix_oneset(&D, &pyramids, SIDELENGTH_FACTOR, FINESTSIDE);

       
  // write the matrix to disk
  fprintf(stderr, "Writing cost matrix to %s...\n", costsOutname);
  writeDoubleMatrix(D, numPointsets, numPointsets, costsOutname);
  



  // delete everything
  int i, j, k;
  for(i=0; i < numPointsets; i++)
    delete[] D[i];
  delete[] D;

  deletePointSets(pointsets, numPointsets);
  

  for(i=0; i < numLevels; i++)
    delete[] translations[i];
  delete[] translations;

  
  for(j=0; j < pyramids.size(); j++){
    for(k=0; k < pyramids[j].size(); k++)
      deleteSparseVec(&pyramids[j][k]);
  }  

  


}
