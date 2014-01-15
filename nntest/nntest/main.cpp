//
//  main.cpp
//  nntest
//
//  Created by Tobias Domhan on 12/24/13.
//  Copyright (c) 2013 Tobias Domhan. All rights reserved.
//

#include <iostream>
#include <cassert>
#include <cmath>
#include <memory>

#include "metrics.h"
#include "data.h"
#include "data_cpu.h"
#include "data_layer.h"
#include "linear_layer.h"
#include "relu_layer.h"
#include "conv_layer.h"
#include "softmax_layer.h"
#include "deepnet.h"
#include "math_util.h"
#include "tests.h"

#include "dataset_cifar10.h"

#define EPS 0.0001

/*
  Use LBFGS library: http://www.chokkan.org/software/liblbfgs/
 */


void network_test_performance(DeepNetwork* network, DataSet* dataset) {
  //Test set performance
  const std::vector<std::unique_ptr<Data>> &test_data = dataset->get_test_data_batches();
  std::vector<std::unique_ptr<Data>>::const_iterator current_test_data = test_data.begin();
  const std::vector<std::unique_ptr<Data>> &test_labels = dataset->get_test_labels_batches();
  std::vector<std::unique_ptr<Data>>::const_iterator current_test_labels = test_labels.begin();
  
  std::vector<std::unique_ptr<Data>> test_predictions;
  
  int subset_count = 1000;
  int subset_idx = 0;
  for (; current_test_data != test_data.end(); current_test_data++, current_test_labels++) {
    network->forward((*current_test_data).get());
    
    std::unique_ptr<Data> probabilities = network->get_output();
    std::unique_ptr<Data> predictions = MaxProbabilityPrediction().execute(probabilities.get());
    test_predictions.push_back(std::move(predictions));
    
    if (subset_idx > subset_count) {
      break;
    }
  }
  std::cout << "Test set accuracy: " << accuracy(test_predictions, test_labels) << std::endl;
  test_predictions.clear();
}



void run_example() {
  int batch_size = 100;
  int num_out = 10;
  
  DataSet* dataset = new DataSetCIFAR10("/Users/tdomhan/Projects/nntest/data/", 1, batch_size);
  
  DeepNetwork* network_ptr = new DeepNetwork(batch_size, dataset->get_data_dimension());
  std::unique_ptr<DeepNetwork> network = std::unique_ptr<DeepNetwork>(network_ptr);
  
  network->add_layer(std::unique_ptr<Layer>(
                                            new ConvLayer(12, 1, 5, 5)
                                            ));
  network->add_layer(std::unique_ptr<Layer>(
                                            new ReluLayer()
                                            ));
/*  network->add_layer(std::unique_ptr<Layer>(
                                            new ConvLayer(24, 1, 5, 5)
                                            ));
  network->add_layer(std::unique_ptr<Layer>(
                                            new ReluLayer()
                                            ));*/
  network->add_layer(std::unique_ptr<Layer>(
                                            new LinearLayer(500)
                                            ));
  network->add_layer(std::unique_ptr<Layer>(
                                            new ReluLayer()
                                            ));
  network->add_layer(std::unique_ptr<Layer>(
                                            new LinearLayer(num_out)
                                            ));
  network->add_layer(std::unique_ptr<Layer>(
                                            new SoftMaxLayer()
                                            ));
  
  for(int epoch=0;epoch<100;epoch++) {
    std::cout << "epoch" << std::endl;
    const std::vector<std::unique_ptr<Data>> &train_data = dataset->get_train_data_batches();
    std::vector<std::unique_ptr<Data>>::const_iterator current_train_data = train_data.begin();
    const std::vector<std::unique_ptr<Data>> &train_labels = dataset->get_train_labels_batches();
    std::vector<std::unique_ptr<Data>>::const_iterator current_train_labels = train_labels.begin();
    
    int current_batch = 1;
    for (; current_train_data != train_data.end(); current_train_data++, current_train_labels++) {
      network->forward((*current_train_data).get());
      network->backward((*current_train_labels).get());
      
      network->update(0.01);
      
      std::cout << current_batch << "/" <<  train_data.size() << std::endl;
      
      if(current_batch  * batch_size % (10000) == 0) {
        network_test_performance(network.get(), dataset);
      }
      
      current_batch++;
    }
  }

  std::cout << "done" << std::endl;
}

int main(int argc, const char * argv[])
{
  run_tests();

  run_example();
  return 0;
}

