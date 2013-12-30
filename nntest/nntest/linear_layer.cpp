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
  if(!has_bottom_layer()) {
    return;
  }
  m_weights = new DataCPU(get_bottom_layer()->get_output_size(1), m_num_hidden);
  m_initialize->execute(m_weights);
  m_bias = new DataCPU(1, m_num_hidden);
  m_initialize->execute(m_bias);
  m_output = new DataCPU(get_bottom_layer()->get_output_size(0), m_num_hidden);
}

//Forward pass
void LinearLayer::forward() {
  if(!has_bottom_layer()) {
    return;
  }
  Data* out_bottom = get_bottom_layer()->get_output();
  std::cout << "bottom begin" << std::endl;
  out_bottom->print();
  std::cout << "bottom end" << std::endl;
  MatrixMultiplicationMKL matrix_multiplication(
                                             out_bottom,
                                             m_weights,
                                             m_output);
  matrix_multiplication.execute();
  //add bias to each batch
  PlusEqualRow().execute(m_output, m_bias);
}


void LinearLayer::backward() {
  
}


Data* LinearLayer::get_output() {
  return m_output;
}

int LinearLayer::get_output_size(int dimension) {
  return m_num_hidden;
}


