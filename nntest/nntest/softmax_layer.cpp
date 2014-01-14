//
//  softmax_layer.cpp
//  nntest
//
//  Created by Tobias Domhan on 12/30/13.
//  Copyright (c) 2013 Tobias Domhan. All rights reserved.
//

#include "softmax_layer.h"

#include <exception>

#include "math_util.h"
#include "data_cpu.h"


SoftMaxLayer::SoftMaxLayer() {
}

SoftMaxLayer::~SoftMaxLayer() {
  delete m_output;
  delete m_backprop_error;
  delete m_total_loss;
}

void SoftMaxLayer::setup() {
  //won't work without a layer below
  assert(has_bottom_layer());
  
  Data* data_bottom = get_bottom_layer()->get_output();
  //we expect flattened data:
  assert(data_bottom->get_num_channels()==1);
  assert(data_bottom->get_height()==1);

  m_output = new DataCPU(data_bottom->get_num_samples(),
                         1,
                         1,
                         data_bottom->get_width());
  m_backprop_error = new DataCPU(data_bottom->get_num_samples(),
                                 1,
                                 1,
                                 data_bottom->get_width());
  

  m_total_loss = new DataCPU(1,
                             1,
                             data_bottom->get_num_samples(),
                             data_bottom->get_width());
}

//Forward pass
void SoftMaxLayer::forward() {
  Data* data_bottom = get_bottom_layer()->get_output();
  m_output->copy_from(*data_bottom);
  SoftmaxRowByRow().execute(m_output);
}


void SoftMaxLayer::backward() {
  throw std::runtime_error("not implemented");
}

// the error
// d/dx softmax(x_i) = [label_i == 1] - p(x_i)
// cost function: NLL
// L = - 1/m forall i [label_i == 1] - p(x_i)
double SoftMaxLayer::backward(Data* expected_output) {
  //TODO: check that m_labels is in one-hot encoding
  
  m_backprop_error->copy_from(*m_output);
  std::unique_ptr<Data> backprop_error_matrix = m_backprop_error->flatten_to_matrix();
  std::unique_ptr<Data> expected_output_matrix = expected_output->flatten_to_matrix();

  MatrixAdd(-1).execute(backprop_error_matrix.get(), expected_output_matrix.get());
  //TODO: show we return the loss for each batch separately or just reduce it to the average?.
  
  m_total_loss->copy_from(*m_output);
  MatrixLog().execute(m_total_loss);
  
  //mask to get only the probabilities of the true labels:
  MatrixElementwiseMultiplication(m_total_loss, expected_output_matrix.get(), m_total_loss).execute();
  
  long num_batches = m_output->get_num_samples();
  double nll = -1./((float)num_batches)*DataSum().execute(m_total_loss);
  
  std::cout << "NLL: " << nll << std::endl;
  return nll;
}

//output off this layer after the forward pass
Data* SoftMaxLayer::get_output() {
  return m_output;
}
