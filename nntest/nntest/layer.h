//
//  layer.h
//  nntest
//
//  Created by Tobias Domhan on 12/28/13.
//  Copyright (c) 2013 Tobias Domhan. All rights reserved.
//

#ifndef nntest_layer_h
#define nntest_layer_h

#include "data.h"

class Layer {
public:
  Layer();

  void connect_top(Layer* top) {m_top = top; m_has_top = true;};
  
  void connect_bottom(Layer* bottom) {m_bottom = bottom; m_has_bottom = true;};
  
  virtual void setup() = 0;
  
  //Forward pass
  virtual void forward() = 0;
  
  // the error
  virtual void backward() = 0;
  
  virtual void update(double learning_rate) = 0;
  
  //output off this layer afer the forward pass
  virtual Data* get_output() = 0;
  
  virtual Data* get_backprop_error() = 0;
  
  virtual int get_output_size(int dimension) = 0;
 
  Layer* get_bottom_layer() {return m_bottom;};
  Layer* get_top_layer() {return m_top;};
  
  bool has_bottom_layer() {return m_has_bottom;};
  bool has_top_layer() {return m_has_top;};
  
private:
  Layer* m_bottom;
  Layer* m_top;
  bool m_has_bottom;
  bool m_has_top;
};


#endif
