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
#include "softmax_layer.h"
#include "deepnet.h"
#include "math_util.h"

#include "dataset_cifar10.h"

#define EPS 0.0001


void network_test_performance(DeepNetwork* network, DataSet* dataset) {
  //Test set performance
  const std::vector<Data*> &test_data = dataset->get_test_data_batches();
  std::vector<Data*>::const_iterator current_test_data = test_data.begin();
  const std::vector<Data*> &test_labels = dataset->get_test_labels_batches();
  std::vector<Data*>::const_iterator current_test_labels = test_labels.begin();
  
  std::vector<std::unique_ptr<Data>> test_predictions;
  std::vector<Data*> test_predictions_raw;
  
  for (; current_test_data != test_data.end(); current_test_data++, current_test_labels++) {
    network->forward(*current_test_data);
    
    std::unique_ptr<Data> probabilities = network->get_output();
    std::unique_ptr<Data> predictions = MaxProbabilityPrediction().execute(probabilities.get());
    test_predictions.push_back(std::move(predictions));
    test_predictions_raw.push_back(test_predictions.back().get());
  }
  std::cout << "Test set accuracy: " << accuracy(test_predictions_raw, test_labels) << std::endl;
  test_predictions.clear();
}

void test_matrix_multiplication() {
  double d1[4][2] = {{1,2},{3,4},{5,6},{7,8}};
  double d2[2][4] = {{1,2,3,4},{5,6,7,8}};
  DataCPU m1(4,2, (double*)d1);
  DataCPU m2(2,4, (double*)d2);
  std::cout << "m1" << std::endl;
  m1.print();
  std::cout << "m2" << std::endl;
  m2.print();
  DataCPU result(4,4);
  MatrixMultiplicationMKL mm(&m1, &m2, &result);
  mm.execute();
  std::cout << "result" << std::endl;
  result.print();
  assert(fabs(result.get_data()[0] - (1*1+2*5)) < EPS);
  assert(fabs(result.get_data()[result.get_count()-1] - (7*4 + 8*8)) < EPS);
}

void test_matrix_multiplication_transpose() {
  std::cout << "Matrix transposes test:" << std::endl;
  /*
    Multiplying by hand:
              1   5
              2   6
              3   7
              4   8
   
    1 3 5 7  50 114
    2 4 6 8  60 140
   */
  double d1[4][2] = {{1,2},{3,4},{5,6},{7,8}};
  double d2[2][4] = {{1,2,3,4},{5,6,7,8}};
  DataCPU m1(4,2, (double*)d1);
  DataCPU m2(2,4, (double*)d2);
  std::cout << "m1" << std::endl;
  m1.print();
  std::cout << "m2" << std::endl;
  m2.print();
  DataCPU result(2,2);
  MatrixMultiplicationMKL mm(&m1, &m2,&result,
                             MatrixMultiplication::MatrixOp::MatrixTranspose,
                             MatrixMultiplication::MatrixOp::MatrixTranspose);
  mm.execute();
  std::cout << "result" << std::endl;
  result.print();
  assert(fabs(result.get_data()[0] - (1*1+3*2+5*3+7*4)) < EPS);
  assert(fabs(result.get_data()[result.get_count()-1] - (2*5+4*6+6*7+8*8)) < EPS);
}


void test_layer(Layer* layer, Data* input, Data* expected_output) {
  DataLayer* data_layer = new DataLayer(input);

  data_layer->connect_top(layer);
  layer->connect_bottom(data_layer);
  
  data_layer->setup();
  layer->setup();
  
  data_layer->forward();
  layer->forward();
  
  Data* output = layer->get_output();
  assert(output->get_size_dim(0) == expected_output->get_size_dim(0));
  assert(output->get_size_dim(1) == expected_output->get_size_dim(1));
  
  int out_size_dim0 = layer->get_output_size(0);
  int out_size_dim1 = layer->get_output_size(1);
  for (int i=0; i<out_size_dim0; i++) {
    for (int j=0; j<out_size_dim1; j++) {
      assert((output->get_data_at(i, j) - expected_output->get_data_at(i, j)) < EPS);
    }
  }
  
  delete data_layer;
}

void test_linear_layer() {
  int num_hidden = 10;
  LinearLayer* linear_layer = new LinearLayer(num_hidden, new SetConst(1));
  int num_samples = 20;
  int input_dim = 5;
  Data* data = new DataCPU(num_samples, input_dim);
  SetConst(1).execute(data);
  data->print();
  DataLayer* data_layer = new DataLayer(data);
  
  data_layer->connect_top(linear_layer);
  linear_layer->connect_bottom(data_layer);
  
  data_layer->setup();
  linear_layer->setup();
  
  data_layer->forward();
  linear_layer->forward();
  
  
  Data* output = linear_layer->get_output();
  int out_size_dim0 = linear_layer->get_output_size(0);
  int out_size_dim1 = linear_layer->get_output_size(1);
  for (int i=0; i<out_size_dim0; i++) {
    for (int j=0; j<out_size_dim1; j++) {
      assert(output->get_data()[output->get_index(i,j)] == input_dim + 1);
    }
  }
  
  delete linear_layer;
  delete data_layer;
  std::cout << "Completed linear layer test" << std::endl;
}

void test_relu_layer() {
  ReluLayer* relu_layer = new ReluLayer();
  int num_samples = 2;
  int input_dim = 2;
  double d_in[2][2] = {{1,-2},{0,5}};
  double d_out_expected[2][2] = {{1,0},{0,5}};
  Data* data_input = new DataCPU(num_samples, input_dim, (double*)d_in);
  Data* data_expected_output = new DataCPU(num_samples, input_dim, (double*)d_out_expected);
  
  test_layer(relu_layer, data_input, data_expected_output);
  
  delete relu_layer;
  delete data_input;
  delete data_expected_output;
  std::cout << "Completed relu layer test" << std::endl;
}


void test_softmax_layer() {
  SoftMaxLayer* soft_max = new SoftMaxLayer();
  int num_samples = 2;
  int input_dim = 2;
  double d_in[2][2] = {{1,9},{0,5}};
  double d_out_expected[2][2] = {{exp(1)/(exp(1)+exp(9)), exp(9)/(exp(1)+exp(9))},
                                 {exp(0)/(exp(0)+exp(5)), exp(5)/(exp(0)+exp(5))}};
  Data* data_input = new DataCPU(num_samples, input_dim, (double*)d_in);
  Data* data_expected_output = new DataCPU(num_samples, input_dim, (double*)d_out_expected);
  
  test_layer(soft_max, data_input, data_expected_output);
  
  delete soft_max;
  delete data_input;
  delete data_expected_output;
  std::cout << "Completed softmax layer test" << std::endl;
}

void connect(Layer* bottom, Layer* top) {
  bottom->connect_top(top);
  top->connect_bottom(bottom);
}

void run_example() {
  int batch_size = 100;
  int num_out = 10;
  
  DataSet* dataset = new DataSetCIFAR10("/Users/tdomhan/Projects/nntest/data/", 1, batch_size);
  
  DeepNetwork* network_ptr = new DeepNetwork(batch_size, dataset->get_data_dimension());
  std::unique_ptr<DeepNetwork> network = std::unique_ptr<DeepNetwork>(network_ptr);
  
  network->add_layer(std::unique_ptr<Layer>(
                                            new LinearLayer(1000)
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
  network->setup();
  
  for(int epoch=0;epoch<100;epoch++) {
    std::cout << "epoch" << std::endl;
    const std::vector<Data*> &train_data = dataset->get_train_data_batches();
    std::vector<Data*>::const_iterator current_train_data = train_data.begin();
    const std::vector<Data*> &train_labels = dataset->get_train_labels_batches();
    std::vector<Data*>::const_iterator current_train_labels = train_labels.begin();
    
    int current_batch = 1;
    for (; current_train_data != train_data.end(); current_train_data++, current_train_labels++) {
      network->forward(*current_train_data);
      network->backward(*current_train_labels);
      
      network->update(0.05);
      
      std::cout << current_batch << "/" <<  train_data.size() << std::endl;
      
      if(current_batch % 10 == 0) {
        network_test_performance(network.get(), dataset);
      }
      
      current_batch++;
    }
  }

  std::cout << "done" << std::endl;
}

int main(int argc, const char * argv[])
{
  test_matrix_multiplication();
  test_matrix_multiplication_transpose();

  test_linear_layer();
  test_relu_layer();
  test_softmax_layer();

  run_example();
  return 0;
}

