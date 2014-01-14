//
//  relu_layer.h
//  nntest
//
//  Created by Tobias Domhan on 12/30/13.
//  Copyright (c) 2013 Tobias Domhan. All rights reserved.
//

#ifndef __nntest__relu_layer__
#define __nntest__relu_layer__

#include <iostream>

#include "layer.h"

#include <memory>

class ReluLayer : public Layer {
public:
  ReluLayer();
  
  ~ReluLayer();
  
  virtual void setup();
  
  //Forward pass
  virtual void forward();
  
  // the error
  virtual void backward();
  
  virtual void update(double learning_rate) {};
  
  //output off this layer afer the forward pass
  virtual Data* get_output();
  
  virtual Data* get_backprop_error();
  
private:
  std::unique_ptr<Data> m_output;
  std::unique_ptr<Data> m_backprop_error;
};

#endif /* defined(__nntest__relu_layer__) */
