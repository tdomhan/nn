//
//  softmax_layer.cpp
//  nntest
//
//  Created by Tobias Domhan on 12/30/13.
//  Copyright (c) 2013 Tobias Domhan. All rights reserved.
//

#include "softmax_layer.h"

#include "math_util.h"
#include "data_cpu.h"


SoftMaxLayer::SoftMaxLayer() : m_labels(NULL) {
}

SoftMaxLayer::~SoftMaxLayer() {
  delete m_output;
  delete m_backprop_error;
  delete m_total_loss;
}

void SoftMaxLayer::setup() {
  //won't work without a layer below
  assert(has_bottom_layer());
  
  m_output = new DataCPU(get_bottom_layer()->get_output_size(0),
                         get_bottom_layer()->get_output_size(1));
  m_backprop_error = new DataCPU(get_bottom_layer()->get_output_size(0),
                                 get_bottom_layer()->get_output_size(1));
  
  m_total_loss = new DataCPU(get_bottom_layer()->get_output_size(0),
                             get_bottom_layer()->get_output_size(1));
}

//Forward pass
void SoftMaxLayer::forward() {
  Data* data_bottom = get_bottom_layer()->get_output();
  m_output->copy_from(*data_bottom);
  SoftmaxRowByRow().execute(m_output);
}

// the error
// d/dx softmax(x_i) = [label_i == 1] - p(x_i)
// cost function: NLL
// L = - 1/m forall i [label_i == 1] - p(x_i)
void SoftMaxLayer::backward() {
  assert(m_labels);
  //TODO: check that m_labels is in one-hot encoding
  
  m_backprop_error->copy_from(*m_output);
  MatrixAdd(-1).execute(m_backprop_error, m_labels);
  //TODO: show we return the loss for each batch separately or just reduce it to the average?.
  
  m_total_loss->copy_from(*m_output);
  MatrixLog().execute(m_total_loss);
  
  //mask to get only the probabilities of the true labels:
  MatrixElementwiseMultiplication(m_total_loss, m_labels, m_total_loss).execute();
  
  long num_batches = m_output->get_size_dim(0);
  double nll = -1./((float)num_batches)*MatrixSum().execute(m_total_loss);
  
  std::cout << "NLL: " << nll << std::endl;
}

//output off this layer after the forward pass
Data* SoftMaxLayer::get_output() {
  return m_output;
}

int SoftMaxLayer::get_output_size(int dimension) {
  return get_bottom_layer()->get_output_size(dimension);
}

void SoftMaxLayer::set_labels(Data* labels) {
  assert(labels->get_size_dim(0) == m_output->get_size_dim(0));
  assert(labels->get_size_dim(1) == m_output->get_size_dim(1));
  
  m_labels = labels;
}

std::unique_ptr<Data> SoftMaxLayer::get_predictions() {
  std::unique_ptr<Data> predictions(new DataCPU(m_output->get_size_dim(0),
                                                m_output->get_size_dim(1)));
  
  SetConst(0).execute(predictions.get());
  
  long num_rows = m_output->get_size_dim(0);
  long num_columns = m_output->get_size_dim(0);
  for (long row=0; row<num_rows; row++) {
    double max = -1.;
    long max_col = -1;
    for (long column=0; column<num_columns; column++) {
      double val = predictions->get_data_at(row, column);
      if (val > max) {
        max = val;
        max_col = column;
      }
    }
    predictions->get_data()[predictions->get_index(row, max_col)] = 1;
  }
  
  return predictions;
}
