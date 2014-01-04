//
//  relu_layer.cpp
//  nntest
//
//  Created by Tobias Domhan on 12/30/13.
//  Copyright (c) 2013 Tobias Domhan. All rights reserved.
//

#include "relu_layer.h"

#include "math_util.h"
#include "data_cpu.h"


ReluLayer::ReluLayer() {
}

ReluLayer::~ReluLayer() {
  delete m_output;
}

void ReluLayer::setup() {
  //won't work without a layer below
  assert(has_bottom_layer());
  m_output = new DataCPU(get_bottom_layer()->get_output_size(0),
                         get_bottom_layer()->get_output_size(1));
  m_backprop_error = new DataCPU(m_output->get_size_dim(0),
                                 m_output->get_size_dim(1));
}

//Forward pass
void ReluLayer::forward() {
  assert(has_bottom_layer());
  
  Data* data_bottom = get_bottom_layer()->get_output();
  m_output->copy_from(*data_bottom);
  AllNegativeZero().execute(m_output);
}

/*
 * f'(x) = (x > 0)
 */
void ReluLayer::backward() {
  assert(has_top_layer());
  assert(has_bottom_layer());
  
  //TODO: we could temporarily store this somewhere else
  Data* data_bottom = get_bottom_layer()->get_output();
  Data* backprop_error_top = get_top_layer()->get_backprop_error();
  assert(data_bottom->get_size_dim(0) == get_backprop_error()->get_size_dim(0));
  assert(data_bottom->get_size_dim(1) == get_backprop_error()->get_size_dim(1));
  m_backprop_error->copy_from(*backprop_error_top);
  AllNegativeZeroMasked().execute(m_backprop_error, data_bottom);
}

//output off this layer after the forward pass
Data* ReluLayer::get_output() {
  return m_output;
}

Data* ReluLayer::get_backprop_error() {
  return m_backprop_error;
};

int ReluLayer::get_output_size(int dimension) {
  return get_bottom_layer()->get_output_size(dimension);
}
