//
//  conv_layer.h
//  nntest
//
//  Created by Tobias Domhan on 1/13/14.
//  Copyright (c) 2014 Tobias Domhan. All rights reserved.
//

#ifndef __nntest__conv_layer__
#define __nntest__conv_layer__

#include <iostream>
#include "layer.h"

#include "math_util.h"
#include "im2col.h"

#include <memory>

class ConvLayer : public Layer {
public:
  ConvLayer(int num_filters,
            int stride,
            int filter_height,
            int filter_width);
  
  ConvLayer(int num_filters,
            int stride,
            int filter_height,
            int filter_width,
            UnaryMathOp const* initialize);
  
  ~ConvLayer();
  
  virtual void setup();
  
  //Forward pass
  virtual void forward();
  
  // the error
  virtual void backward();
  
  //update the weights
  virtual void update(double learning_rate);
  
  //output off this layer afer the forward pass
  virtual Data* get_output() {return m_output.get();};
  
  virtual Data* get_backprop_error() {return m_backprop_error.get();};
  
  //virtual int get_output_size(int dimension);
  
private:
  void initialize_weights();
  void initialize_bias();
  
  int m_num_filters;
  int m_stride;
  int m_filter_height;
  int m_filter_width;
  
  Im2Col im2col;
  
  std::unique_ptr<Data> m_weights;
  std::unique_ptr<Data> m_weights_update;
  
  std::unique_ptr<Data> m_bias;
  std::unique_ptr<Data> m_bias_update;
  
  std::unique_ptr<Data> m_output;
  //matrix view of the output data
  std::unique_ptr<Data> m_output_matrix;
  std::unique_ptr<Data> m_backprop_error;
  
  std::unique_ptr<Data> m_col_image;
  
  UnaryMathOp const* m_initialize;
};

#endif /* defined(__nntest__conv_layer__) */
