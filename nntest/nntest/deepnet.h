//
//  net.h
//  nntest
//
//  Created by Tobias Domhan on 1/1/14.
//  Copyright (c) 2014 Tobias Domhan. All rights reserved.
//

#ifndef __nntest__net__
#define __nntest__net__

#include <iostream>

#include "layer.h"

#include <vector>
#include <memory>

class DeepNetwork {
public:
  
  DeepNetwork(int batch_size, int data_dimension);
  
  void set_input(Data* input);
  
  std::unique_ptr<Data> get_output();
  
  //setup all the layers
  //must be called before using the network
  void setup();
  
  void forward();
  
  void backward();
  
  void update(double learning_rate);
  
  //add another layer on top
  void add_layer(std::unique_ptr<Layer> layer);

private:
  void check_setup();
  
  std::vector<std::unique_ptr<Layer>> m_layers;
  
  bool m_setup;
};

#endif /* defined(__nntest__net__) */
