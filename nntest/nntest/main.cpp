//
//  main.cpp
//  nntest
//
//  Created by Tobias Domhan on 12/24/13.
//  Copyright (c) 2013 Tobias Domhan. All rights reserved.
//

#include <iostream>
#include <cassert>

#include "data.h"
#include "data_cpu.h"
#include "data_layer.h"
#include "linear_layer.h"

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
  assert(result.get_data()[0] == 1*1+2*5);
  assert(result.get_data()[result.get_count()-1] == 7*4 + 8*8);
}

void test_linear_layer() {
  int num_hidden = 10;
  LinearLayer* linear_layer = new LinearLayer(num_hidden, new SetConst(1));
  int num_samples = 20;
  int input_dim = 5;
  Data* data = new DataCPU(num_samples, input_dim);
  SetConst(1).execute(data);
  data->print();
  DataLayer* data_layer = new DataLayer(data, num_samples);
  data_layer->connect_top(linear_layer);
  linear_layer->connect_bottom(data_layer);
  
  data_layer->setup();
  linear_layer->setup();
  
  data_layer->forward();
  linear_layer->forward();
  
  
  Data* output = linear_layer->get_output();
    std::cout << "linear layer out begind" << std::endl;
  output->print();
    std::cout << "linear layer out end" << std::endl;
  int out_size_dim0 = linear_layer->get_output_size(0);
  int out_size_dim1 = linear_layer->get_output_size(1);
  for (int i=0; i<out_size_dim0; i++) {
    for (int j=0; j<out_size_dim1; j++) {
      std::cout << output->get_data()[output->get_index(i,j)] << std::endl;
      assert(output->get_data()[output->get_index(i,j)] == input_dim + 1);
    }
  }
  
  delete linear_layer;
  delete data_layer;
  std::cout << "Completed linear layer test" << std::endl;
}


void run_example() {
  int num_samples = 100;
  int input_dim = 5;
  Data* data = new DataCPU(num_samples, input_dim);
  double* d = data->get_data();
  for(int i=0;i<num_samples;i++) {
    for(int j=0; j<input_dim; j++) {
      d[i*input_dim+j] = i;
    }
  }
  DataLayer* data_layer = new DataLayer(data, 10);
  
  LinearLayer* linear_layer = new LinearLayer(15);
  
  data_layer->connect_top(linear_layer);
  linear_layer->connect_bottom(data_layer);
  
  data_layer->setup();
  linear_layer->setup();
  
  data_layer->forward();
  linear_layer->forward();
  
  linear_layer->get_output()->print();
  
  std::cout << "done1" << std::endl;
  
  while(data_layer->batches_remaining()){
    data_layer->forward();
    linear_layer->forward();
    
    std::cout << "batch" << std::endl;
    data_layer->next_batch();
  }
  
  std::cout << "done" << std::endl;
  delete linear_layer;
  delete data_layer;
  delete data;
}

int main(int argc, const char * argv[])
{
  test_matrix_multiplication();
  test_linear_layer();

  run_example();
  return 0;
}

