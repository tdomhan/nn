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

class ReluLayer : public Layer {
public:
  ReluLayer();
  
  ~ReluLayer();
  
  virtual void setup();
  
  //Forward pass
  virtual void forward();
  
  // the error
  virtual void backward();
  
  //output off this layer afer the forward pass
  virtual Data* get_output();
  
  virtual int get_output_size(int dimension);
  
private:
  Data* m_output;
};

#endif /* defined(__nntest__relu_layer__) */
