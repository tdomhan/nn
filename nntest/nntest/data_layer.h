//
//  data_layer.h
//  nntest
//
//  Created by Tobias Domhan on 12/28/13.
//  Copyright (c) 2013 Tobias Domhan. All rights reserved.
//

#ifndef nntest_data_layer_h
#define nntest_data_layer_h

#include "layer.h"
#include "data.h"

/**
 * Provides input data in batches.
 */
class DataLayer : public Layer {
public:
  DataLayer(Data* output);
  
  virtual void setup();
  
  virtual void forward();
  
  virtual void backward();
  
  virtual void update(double learning_rate) {};

  virtual Data* get_output();
  
  virtual Data* get_backprop_error() {return NULL;};
  
  virtual int get_output_size(int dimension) {return m_output->get_size_dim(dimension);};
  
  void set_current_output(Data* output);
  
private:
  Data* m_output;
};

#endif
