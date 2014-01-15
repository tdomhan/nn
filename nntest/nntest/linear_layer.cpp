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
    //m_initialize(new UniformRandom(0.05))
      m_initialize(new FanInScaleFiller())
{

}

LinearLayer::LinearLayer(int num_hidden, UnaryMathOp const* initialize)
: m_num_hidden(num_hidden),
  m_initialize(initialize)
{
  
}

LinearLayer::~LinearLayer() {
  delete m_weights;
  delete m_weights_update;
  delete m_bias;
  delete m_bias_update;
  delete m_batch_average_vector;
  delete m_output;
  delete m_backprop_error;
  delete m_initialize;
}

void LinearLayer::setup() {
  assert(has_bottom_layer());
  
  initialize_weights();
  
  initialize_bias();
  
  m_batch_average_vector = new DataCPU(1, get_bottom_layer()->get_output()->get_num_samples());
  SetConst(1).execute(m_batch_average_vector);
  
  m_output = new DataCPU(get_bottom_layer()->get_output()->get_num_samples(),
                         1,
                         1,
                         m_num_hidden);
  
  //provide the error to the layer below
  m_backprop_error = new DataCPU(get_bottom_layer()->get_output()->get_num_samples(),
                                 1,
                                 1,
                                 get_bottom_layer()->get_output()->get_count_per_sample());
}

void LinearLayer::initialize_weights() {
  m_weights = new DataCPU(get_bottom_layer()->get_output()->get_count_per_sample(),
                          m_num_hidden);
  m_weights_update = new DataCPU(m_weights->get_height(),
                                 m_weights->get_width());
  m_initialize->execute(m_weights);
}

void LinearLayer::initialize_bias() {
  m_bias = new DataCPU(1, m_num_hidden);
  m_bias_update = new DataCPU(1, m_num_hidden);
  m_initialize->execute(m_bias);
}

//Forward pass
void LinearLayer::forward() {
  assert(has_bottom_layer());
  Data* out_bottom = get_bottom_layer()->get_output();
//  std::cout << "bottom begin" << std::endl;
//  out_bottom->print();
//  std::cout << "bottom end" << std::endl;
  //TODO: do multiplication in one GEMM step?
  
  //flatten any data dimensions to a single dimension
  std::unique_ptr<Data> out_bottom_flat = out_bottom->flatten_to_matrix();
  std::unique_ptr<Data> output_flat = m_output->flatten_to_matrix();

  //linear combination of the inputs
  MatrixMultiplicationMKL matrix_multiplication(
                                             out_bottom_flat.get(),
                                             m_weights,
                                             output_flat.get());
  matrix_multiplication.execute();
  //add bias to each batch
  PlusEqualRow().execute(output_flat.get(), m_bias);
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
 * f'(x)/d b_j  = error_j
 */
void LinearLayer::backward() {
  assert(has_top_layer());

  //m x k
  Data* backprop_error_top = get_top_layer()->get_backprop_error();
  std::unique_ptr<Data> backprop_error_top_flat = backprop_error_top->flatten_to_matrix();
  
  // m x n
  Data* out_bottom = get_bottom_layer()->get_output();
  std::unique_ptr<Data> out_bottom_flat = out_bottom->flatten_to_matrix();

  //gradient given weights
  double batch_average = 1. / out_bottom->get_num_samples();
  MatrixMultiplicationMKL calculate_weight_gradient(out_bottom_flat.get(),
                                                    backprop_error_top_flat.get(),
                                                    m_weights_update,
                                                    MatrixMultiplication::MatrixOp::MatrixTranspose,
                                                    MatrixMultiplication::MatrixOp::NoTranspose,
                                                    batch_average);
  calculate_weight_gradient.execute();

  //gradient given the bias
  MatrixMultiplicationMKL calculate_bias_gradient(m_batch_average_vector,
                                                  backprop_error_top_flat.get(),
                                                  m_bias_update,
                                                  MatrixMultiplication::MatrixOp::NoTranspose,
                                                  MatrixMultiplication::MatrixOp::NoTranspose,
                                                  batch_average);
  calculate_bias_gradient.execute();
  
  std::unique_ptr<Data> backprop_error_flat = m_backprop_error->flatten_to_matrix();
  
  //gradient given input from layer below
  MatrixMultiplicationMKL calculate_error_given_input(backprop_error_top_flat.get(),
                                                      m_weights,
                                                      backprop_error_flat.get(),
                                                      MatrixMultiplication::MatrixOp::NoTranspose,
                                                      MatrixMultiplication::MatrixOp::MatrixTranspose,
                                                      batch_average);
  calculate_error_given_input.execute();
}

void LinearLayer::update(double learning_rate) {
  MatrixAdd apply_update(-1*learning_rate);
  apply_update.execute(m_weights, m_weights_update);
  apply_update.execute(m_bias, m_bias_update);
  
  std::cout << "Sum of weight updates linear layer" << DataAbsSum().execute(m_weights_update) << std::endl;
}

