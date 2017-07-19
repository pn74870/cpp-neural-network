#pragma once
#include <cstdlib>
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <math.h>

using namespace std;
using namespace Eigen;


typedef Matrix<float, 10, 1> OutVec;

class Network {
private:
	int nLayers;
	vector<int> sizes;
	vector<double> weights;
	vector<double> biases;
	void updateMiniBatch();
	void backDrop();
	OutVec costDerivative(OutVec output, OutVec y );
	double sigmoid(double x);
	double sigmoidPrime(double x);
public:
	Network(vector<int> sizes);
	void feedForward();
	void SGD(vector<double> trainingData,double eta);
	void evaluate();

};