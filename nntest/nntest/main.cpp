//
//  main.cpp
//  nntest
//
//  Created by Tobias Domhan on 12/24/13.
//  Copyright (c) 2013 Tobias Domhan. All rights reserved.
//

#include <iostream>

#include "data.h"
#include "data_cpu.h"
#include "data_layer.h"
#include "linear_layer.h"

int main(int argc, const char * argv[])
{
  int num_samples = 100;
  int input_dim = 5;
  Data* data = new DataCPU(num_samples, input_dim);
  double* d = data->get_data();
  for(int i=0;i<num_samples;i++) {
    for(int j=0; j<input_dim; j++) {
      d[i*input_dim+j] = i;
    }
  }
  DataLayer* data_layer = new DataLayer(data, 10);
  
  LinearLayer* linear_layer = new LinearLayer(15);
  
  data_layer->connect_top(linear_layer);
  linear_layer->connect_bottom(data_layer);
  
  data_layer->setup();
  linear_layer->setup();
  
  data_layer->forward();
  linear_layer->forward();
  
  linear_layer->get_output()->print();
  
  std::cout << "done1" << std::endl;
  
  while(data_layer->batches_remaining()){
    data_layer->forward();
    linear_layer->forward();
    
    std::cout << "batch" << std::endl;
    data_layer->next_batch();
  }
  
  std::cout << "done" << std::endl;
  delete linear_layer;
  delete data_layer;
  delete data;
  return 0;
}

