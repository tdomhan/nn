//
//  linear_layer.h
//  nntest
//
//  Created by Tobias Domhan on 12/28/13.
//  Copyright (c) 2013 Tobias Domhan. All rights reserved.
//

#ifndef nntest_linear_layer_h
#define nntest_linear_layer_h

#include "layer.h"
#include "math_util.h"
#include "data.h"


//TODO: make bias optional

class LinearLayer : public Layer {
public:
  LinearLayer(int num_hidden);
  
  LinearLayer(int num_hidden, UnaryMathOp const* initialize);
  
  ~LinearLayer();
  
  virtual void setup();
  
  //Forward pass
  virtual void forward();
  
  // the error
  virtual void backward();
  
  //update the weights
  virtual void update();
  
  //output off this layer afer the forward pass
  virtual Data* get_output();
  
  virtual Data* get_backprop_error() {return NULL;};
  
  virtual int get_output_size(int dimension);
  
private:
  int m_num_hidden;
  Data* m_weights;
  Data* m_weights_update;
  Data* m_bias;
  Data* m_bias_update;
  Data* m_output;
  Data* m_backprop_error;
  UnaryMathOp const* m_initialize;
};


#endif
