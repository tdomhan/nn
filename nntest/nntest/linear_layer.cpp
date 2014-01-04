//
//  linear_layer.cpp
//  nntest
//
//  Created by Tobias Domhan on 12/28/13.
//  Copyright (c) 2013 Tobias Domhan. All rights reserved.
//

#include "linear_layer.h"
#include "math_util.h"

#include "data_cpu.h"

#include <iostream>


LinearLayer::LinearLayer(int num_hidden)
  : m_num_hidden(num_hidden),
    m_initialize(new UniformRandom(0.01))
{

}

LinearLayer::LinearLayer(int num_hidden, UnaryMathOp const* initialize)
: m_num_hidden(num_hidden),
  m_initialize(initialize)
{
  
}

LinearLayer::~LinearLayer() {
  delete m_initialize;
  delete m_weights;
  delete m_bias;
  delete m_output;
}

void LinearLayer::setup() {
  assert(has_bottom_layer());

  m_weights = new DataCPU(get_bottom_layer()->get_output_size(1),
                          m_num_hidden);
  m_weights_update = new DataCPU(m_weights->get_size_dim(0),
                                 m_weights->get_size_dim(1));
  m_initialize->execute(m_weights);
  
  m_bias = new DataCPU(1, m_num_hidden);
  m_bias_update = new DataCPU(1, m_num_hidden);
  m_initialize->execute(m_bias);
  
  m_batch_average_vector = new DataCPU(1, get_bottom_layer()->get_output_size(0));
  
  m_output = new DataCPU(get_bottom_layer()->get_output_size(0),
                         m_num_hidden);
  //provide the error to the layer below
  m_backprop_error = new DataCPU(get_bottom_layer()->get_output_size(0),
                                 get_bottom_layer()->get_output_size(1));
}

//Forward pass
void LinearLayer::forward() {
  assert(has_bottom_layer());
  Data* out_bottom = get_bottom_layer()->get_output();
//  std::cout << "bottom begin" << std::endl;
//  out_bottom->print();
//  std::cout << "bottom end" << std::endl;
  //TODO: do multiplication in one GEMM step?

  //linear combination of the inputs
  MatrixMultiplicationMKL matrix_multiplication(
                                             out_bottom,
                                             m_weights,
                                             m_output);
  matrix_multiplication.execute();
  //add bias to each batch
  PlusEqualRow().execute(m_output, m_bias);
}


/*
 * m: num_batches
 * n: data dimension
 * k: num_units
 
 * D_IN: m x n
 * W: n x k
 * b: 1 x k
 * D_OUT: m x k
 * ERR_TOP: m x k
 * ERR_OUT: m x n

 * f(x)  = D_IN*W
 * f'(x)/d in = ERR_TOP * W^T =
 * f'(x)/d w_ij = in_i * top_diff_j
 * f'(x)/d b_j  =
 */
void LinearLayer::backward() {
  assert(has_top_layer());

  //m x k
  Data* backprop_error_top = get_top_layer()->get_backprop_error();
  
  // m x n
  Data* out_bottom = get_bottom_layer()->get_output();

  //gradient given weights
  //TODO: multiply by 1/num_batches to get the average over batches
  double batch_average = 1. / get_bottom_layer()->get_output_size(0);
  MatrixMultiplicationMKL calculate_weight_gradient(out_bottom,
                                                    backprop_error_top,
                                                    m_weights_update,
                                                    MatrixMultiplication::MatrixOp::MatrixTranspose,
                                                    MatrixMultiplication::MatrixOp::MatrixTranspose,
                                                    batch_average);
  calculate_weight_gradient.execute();

  //gradient given the bias
  SetConst(1.).execute(m_bias_update);
  MatrixMultiplicationMKL calculate_bias_gradient(m_batch_average_vector,
                                                  backprop_error_top,
                                                  m_bias_update,
                                                  MatrixMultiplication::MatrixOp::NoTranspose,
                                                  MatrixMultiplication::MatrixOp::NoTranspose,
                                                  batch_average);
  calculate_bias_gradient.execute();
  
  
  //gradient given input from layer below
  MatrixMultiplicationMKL calculate_error_given_input(backprop_error_top,
                                                      m_weights,
                                                      m_backprop_error,
                                                      MatrixMultiplication::MatrixOp::NoTranspose,
                                                      MatrixMultiplication::MatrixOp::MatrixTranspose,
                                                      batch_average);
  calculate_error_given_input.execute();
}

void LinearLayer::update(double learning_rate) {
  
}


Data* LinearLayer::get_output() {
  return m_output;
}

int LinearLayer::get_output_size(int dimension) {
  return m_num_hidden;
}


