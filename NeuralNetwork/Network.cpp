#include "Network.h"

Network::Network(vector<int> sizes):sizes(sizes) {
	nLayers = sizes.size();
}


double Network::sigmoid(double x) {
	return 1 / (exp(-x) + 1);
}
double Network::sigmoidPrime(double x) {
	return sigmoid(x)*(1-sigmoid(x));
}

void Network :: updateMiniBatch() {}


void Network::backDrop() {
}


OutVec Network::costDerivative(OutVec output, OutVec y){
//cost=.5*(output-y)^2
	return output - y;
}


void Network::feedForward() {
	for (int i = 1; i < nLayers; i++) {

	}
}
void Network::SGD(vector<double> trainingData, double eta) {}
void Network::evaluate() {}
