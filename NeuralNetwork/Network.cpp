#include "Network.h"


Network::Layer:: Layer(Network* net,int nNeur, int nNeurPrev):size(nNeur),sizePrev(nNeurPrev),net(net){
  if(nNeurPrev>0){  
    w=MatrixXd::Random(nNeur,nNeurPrev);
    b=MatrixXd::Random(nNeur,1);
  }
}

void Network::Layer::feedForward(const VectorXd &aPrev ){
  z=w*aPrev+b;
  a=net->sigmoid(z);
}




Network::Network(vector<int> sizes) {
  layers.push_back(Layer(this,sizes[0]));
  for(int i=1;i<sizes.size();i++){
    layers.push_back(Layer(this,sizes[i],sizes[i-1]));

  }
  
	
}

double Network::sigmoid(double x) {
  
	return 1 / (exp(-x) + 1);
}
double Network::sigmoidPrime(double x) {
	return sigmoid(x)*(1-sigmoid(x));
}



VectorXd Network::sigmoid(const VectorXd &x) {
  VectorXd ans(x.rows());

  for(int i=0;i<x.rows();i++){
    ans(i)=sigmoid(x(i));
  }
	return ans;
}

VectorXd Network::sigmoidPrime(const VectorXd &x) {
  VectorXd ans(x.rows());

  for(int i=0;i<x.rows();i++){
    ans(i)=sigmoidPrime(x(i));
  }
	return ans;
}

void Network :: updateMiniBatch( vector<pair<VectorXd,VectorXd> > miniBatch,double eta) {
  for(auto  it = miniBatch.begin();it!=miniBatch.end();it++){
    backProp(it->first,it->second);
    
  }
   for(auto it = layers.begin();it!=layers.end();it++){
     it->w-=eta/miniBatch.size()*it->gradSumW;
     it->b-=eta/miniBatch.size()*it->gradSumB;
    
  }

}

VectorXd Network::feedForward(const VectorXd& x){
  layers[0].a=x;
  for(int i=1;i<nLayers;i++){
    layers[i].feedForward(layers[i-1].a);
  }
  return layers[nLayers-1].a;
}
void Network::backProp(const VectorXd& x,const VectorXd& y) {
  
  feedForward(x);

  layers[nLayers-1].delta=costDerivative(layers[nLayers-1].a,y).array()*sigmoidPrime(layers[nLayers-1].z).array();
  layers[nLayers - 1].gradSumW += layers[nLayers-1].delta*layers[nLayers-2].a.transpose();
  layers[nLayers - 1].gradSumB += (layers[nLayers - 1].delta);
  for(int i=nLayers-2;i>0;i--){
    layers[i].delta=layers[1+i].w.transpose()*(layers[1+i].delta.array()*sigmoidPrime(layers[i].z).array()).matrix();
    layers[i].gradSumW+=(layers[i].delta*layers[i-1].a.transpose());
    layers[i].gradSumB+=(layers[i].delta);
    // nBatches++;
  }
}



VectorXd Network::costDerivative(const VectorXd& output,const VectorXd& y){
//cost=.5*(output-y)^2
	return output - y;
}


void Network::SGD(vector<pair<VectorXd,VectorXd> > trainingData,int epochs,int miniBatchSize, double eta) {
  for(int i=0;i<epochs;i++){
    random_shuffle(trainingData.begin(),trainingData.end());
    for(int j=0;j<trainingData.size();j+=miniBatchSize){
		bool t = j + miniBatchSize < trainingData.size();
		auto end = t ? trainingData.begin()+j+miniBatchSize:trainingData.end();
      updateMiniBatch(vector<pair<VectorXd,VectorXd> >(make_move_iterator(trainingData.begin()+j),make_move_iterator(end)),eta);
    }
    
    cout<<"epoch "<<i<<" completed\n";
  }
}
VectorXd Network::evaluate(const VectorXd &input) {
  VectorXd ans= feedForward(input);
  cout<<"output:\n"<<ans<<endl;
  return ans;
}


