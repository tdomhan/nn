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
}

void ReluLayer::setup() {
  //won't work without a layer below
  assert(has_bottom_layer());
  Data* bottom_out = get_bottom_layer()->get_output();
  m_output = std::unique_ptr<Data>(new DataCPU(bottom_out->get_size_dim(0),
                         bottom_out->get_size_dim(1),
                         bottom_out->get_size_dim(2),
                         bottom_out->get_size_dim(3)));
  m_backprop_error = std::unique_ptr<Data>(new DataCPU(bottom_out->get_size_dim(0),
                                 bottom_out->get_size_dim(1),
                                 bottom_out->get_size_dim(2),
                                 bottom_out->get_size_dim(3)));
}

//Forward pass
void ReluLayer::forward() {
  assert(has_bottom_layer());
  
  Data* data_bottom = get_bottom_layer()->get_output();
  m_output->copy_from(*data_bottom);
  AllNegativeZero().execute(m_output.get());
}

/*
 * f'(x) = (x > 0)
 */
void ReluLayer::backward() {
  assert(has_top_layer());
  assert(has_bottom_layer());
  
  Data* data_bottom = get_bottom_layer()->get_output();
  Data* backprop_error_top = get_top_layer()->get_backprop_error();

  assert(data_bottom->get_size_dim(0) == get_backprop_error()->get_size_dim(0));
  assert(data_bottom->get_size_dim(1) == get_backprop_error()->get_size_dim(1));
  assert(data_bottom->get_size_dim(2) == get_backprop_error()->get_size_dim(2));
  assert(data_bottom->get_size_dim(3) == get_backprop_error()->get_size_dim(3));
  m_backprop_error->copy_from(*backprop_error_top);
  AllNegativeZeroMasked().execute(m_backprop_error.get(), data_bottom);
}

//output off this layer after the forward pass
Data* ReluLayer::get_output() {
  return m_output.get();
}

Data* ReluLayer::get_backprop_error() {
  return m_backprop_error.get();
};
