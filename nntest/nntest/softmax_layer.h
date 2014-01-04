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

#include "layer.h"

/**
 * Apply the softmax function.
 * f(x) = exp(a_i) / sum(exp(a_j), 0, num_input)
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
  
  //output off this layer afer the forward pass
  virtual Data* get_output();
  
  virtual Data* get_backprop_error() {return NULL;};
  
  virtual int get_output_size(int dimension);
  
private:
  Data* m_output;
};

#endif /* defined(__nntest__softmax_layer__) */
