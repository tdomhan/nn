//
//  softmax_layer.h
//  nntest
//
//  Created by Tobias Domhan on 12/30/13.
//  Copyright (c) 2013 Tobias Domhan. All rights reserved.
//

#ifndef __nntest__softmax_layer__
#define __nntest__softmax_layer__

#include <iostream>
#include <memory>

#include "layer.h"

/**
 * Apply the softmax function.
 * f(x) = exp(a_i) / sum(exp(a_j), 0, num_input)
 * And backpropagates the Negative Log Likelihood given the true labels.
 */
class SoftMaxLayer : public Layer {
public:
  SoftMaxLayer();
  
  ~SoftMaxLayer();
  
  virtual void setup();
  
  //Forward pass
  virtual void forward();
  
  // the error
  virtual void backward();
  
  virtual void update(double learning_rate) {};
  
  //output off this layer afer the forward pass
  virtual Data* get_output();
  
  virtual Data* get_backprop_error() {return m_backprop_error;};
  
  virtual int get_output_size(int dimension);
  
  //labels in one-hot encoding
  void set_labels(Data* labels);
  
  //get predictions (in one-hot encoding)
  std::unique_ptr<Data> get_predictions();
  
private:
  Data* m_output;
  Data* m_backprop_error;
  
  Data* m_total_loss;

  Data* m_labels;
};

#endif /* defined(__nntest__softmax_layer__) */
