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


LinearLayer::LinearLayer(int num_hidden) : m_num_hidden(num_hidden)
{

}

void LinearLayer::setup() {
  if(!has_bottom_layer()) {
    return;
  }
  m_weights = new DataCPU(get_bottom_layer()->get_output_size(1), m_num_hidden);
  UniformRandom(m_weights).execute();
  UniformRandom(m_weights).execute();
  m_bias = new DataCPU(1, m_num_hidden);
  UniformRandom(m_bias).execute();
  UniformRandom(m_bias).execute();
  m_output = new DataCPU(get_bottom_layer()->get_output_size(0), m_num_hidden);
}

//Forward pass
void LinearLayer::forward() {
  if(!has_bottom_layer()) {
    return;
  }
  Data* out_bottom = get_bottom_layer()->get_output();
  MatrixMultiplication matrix_multiplication(out_bottom,
                                             m_weights,
                                             m_output);
  matrix_multiplication.execute();
}


void LinearLayer::backward() {
  
}


Data* LinearLayer::get_output() {
  return m_output;
}

int LinearLayer::get_output_size(int dimension) {
  return m_num_hidden;
}


