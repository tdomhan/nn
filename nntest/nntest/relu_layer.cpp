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
  
  m_output = new DataCPU(get_bottom_layer()->get_output_size(0), get_bottom_layer()->get_output_size(1));
}

//Forward pass
void ReluLayer::forward() {
  Data* data_bottom = get_bottom_layer()->get_output();
  m_output->copy_from(*data_bottom);
  AllNegativeZero().execute(m_output);
}

// the error
void ReluLayer::backward() {
  
}

//output off this layer after the forward pass
Data* ReluLayer::get_output() {
  return m_output;
}

int ReluLayer::get_output_size(int dimension) {
  return get_bottom_layer()->get_output_size(dimension);
}
