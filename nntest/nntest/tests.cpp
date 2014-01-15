//
//  tests.cpp
//  nntest
//
//  Created by Tobias Domhan on 1/15/14.
//  Copyright (c) 2014 Tobias Domhan. All rights reserved.
//

#include "tests.h"


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
  assert(fabs(result.get_data()[result.get_total_count()-1] - (7*4 + 8*8)) < EPS);
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
  assert(fabs(result.get_data()[result.get_total_count()-1] - (2*5+4*6+6*7+8*8)) < EPS);
}

void test_im2col() {
  
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
  assert(output->get_width() == expected_output->get_width());
  assert(output->get_height() == expected_output->get_height());
  
  long out_size_dim0 = layer->get_output()->get_num_samples();
  long out_size_dim1 = layer->get_output()->get_width();
  for (int i=0; i<out_size_dim0; i++) {
    for (int j=0; j<out_size_dim1; j++) {
      assert((output->get_data_at(i, 0, 0, j) - expected_output->get_data_at(i, 0, 0, j)) < EPS);
    }
  }
  
  delete data_layer;
}

void test_layer_gradient() {
//  std::unique_ptr<>
}

void test_linear_layer() {
  int num_hidden = 10;
  LinearLayer* linear_layer = new LinearLayer(num_hidden, new SetConst(1));
  int num_samples = 20;
  int input_dim = 5;
  Data* data = new DataCPU(num_samples, 1, 1, input_dim);
  SetConst(1).execute(data);
  //  data->print();
  DataLayer* data_layer = new DataLayer(data);
  
  data_layer->connect_top(linear_layer);
  linear_layer->connect_bottom(data_layer);
  
  data_layer->setup();
  linear_layer->setup();
  
  data_layer->forward();
  linear_layer->forward();
  
  
  Data* output = linear_layer->get_output();
  long out_size_dim0 = linear_layer->get_output()->get_height();
  long out_size_dim1 = linear_layer->get_output()->get_width();
  for (int i=0; i<out_size_dim0; i++) {
    for (int j=0; j<out_size_dim1; j++) {
      assert(output->get_data()[output->get_index(i,0,0,j)] == input_dim + 1);
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
  Data* data_input = new DataCPU(num_samples, 1, 1, input_dim, (double*)d_in);
  Data* data_expected_output = new DataCPU(num_samples, 1, 1, input_dim, (double*)d_out_expected);
  
  test_layer(soft_max, data_input, data_expected_output);
  
  delete soft_max;
  delete data_input;
  delete data_expected_output;
  std::cout << "Completed softmax layer test" << std::endl;
}


void run_tests() {
  test_matrix_multiplication();
  test_matrix_multiplication_transpose();
  
  test_im2col();
  
  test_linear_layer();
  test_relu_layer();
  test_softmax_layer();
}
