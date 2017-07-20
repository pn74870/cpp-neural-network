#pragma once
#include <cstdlib>
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <math.h>

using namespace std;
using namespace Eigen;




class Network {
private:
	struct Layer{
	  MatrixXd w;
	  VectorXd a;
	  VectorXd b;
	  VectorXd z;
	  VectorXd delta;
	  int size;
	  int sizePrev;
	  MatrixXd gradSumW;
	  VectorXd gradSumB;
	  int nBatches;
	  Network* net;
	  Layer(Network* net,int nNeur, int nNeurPrev=0);
	  
	  void feedForward(const VectorXd& aPrev );
	 
	};
	int nLayers;
	vector<Layer> layers;
	
	void updateMiniBatch( vector<pair<VectorXd,VectorXd> > miniBatch,double eta);
	void backProp(const VectorXd& x,const VectorXd& y);

	VectorXd costDerivative(const VectorXd &output, const VectorXd &y );

	VectorXd sigmoid(const VectorXd &x);
	double sigmoid(double x);
	VectorXd sigmoidPrime(const VectorXd &x);
	double sigmoidPrime(double x);





public:
Network(vector<int> sizes);
VectorXd feedForward(const VectorXd& x);
void SGD( vector<pair<VectorXd,VectorXd> > trainingData,int epochs,int miniBatchSize, double eta);
VectorXd evaluate(const VectorXd &input);

};
