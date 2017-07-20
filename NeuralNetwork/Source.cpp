#include <cstdlib>
#include <Eigen/Dense>
#include <iostream>
#include "Network.h"
#include <vector>
#include <fstream>
#include <iterator>
#include <algorithm>
#include "MNISTParser.h"
using namespace std;
using namespace Eigen;




int main() {
  vector<int> sizes;
  sizes.push_back(784);
  sizes.push_back(30);
  sizes.push_back(10);
  //Network net(sizes);
  
  string dataFile="data/train-images.idx3-ubyte";
  string labelFile = "data/train-labels.idx1-ubyte";
  MNISTDataset mnist;
  assert(0 == mnist.Parse(dataFile.c_str(), labelFile.c_str()));
  const float* imageBuffer=mnist.GetImageData();

  vector<pair<VectorXd, VectorXd>> input;
	
  for(int a=0;a<mnist.GetImageCount();a++)
  for (size_t j = 0; j < 28; ++j)
  {
	  for (size_t i = 0; i < 28; ++i)
	  {
		  printf("%3d ", (uint8_t)imageBuffer[j * 28 + i]);
	  }
	  
  }
  //MatrixXd input(28, 28);
  //ifstream fstream(dataFile, ios::in | ios::binary);


  
  //cout << input;
   

 
    
   


    //net.SDG(const vector<pair<VectorXd,VectorXd> >& trainingData,int epochs,int miniBatchSize, double eta)
  return 0;

}
