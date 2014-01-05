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
  DataLayer(Data* dataset, int batch_size);
  
  virtual void setup();
  
  virtual void forward();
  
  virtual void backward();
  
  virtual void update(double learning_rate) {};

  virtual Data* get_output();
  
  virtual Data* get_backprop_error() {return NULL;};
  
  virtual int get_output_size(int dimension);
  
  void next_batch();
  
  bool batches_remaining();
private:
  Data* m_dataset;
  int m_batch_size;
  int m_current_pointer;
  int m_rows;
  Data* m_output;
};

#endif
