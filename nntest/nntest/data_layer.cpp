//
//  data_layer.cpp
//  nntest
//
//  Created by Tobias Domhan on 12/28/13.
//  Copyright (c) 2013 Tobias Domhan. All rights reserved.
//

#include "data_layer.h"

#include <iostream>

#include "data_cpu.h"

/*
 TODO: move the iteration over the dataset outside of the layer into a separate object.
 */


DataLayer::DataLayer(int batch_size, int data_dimension)
  : m_data(NULL)
{
  output_size[0] = batch_size;
  output_size[1] = data_dimension;
};

DataLayer::DataLayer(Data* data)
 : m_data(data) {
   output_size[0] = data->get_size_dim(0);
   output_size[1] = data->get_size_dim(1);
}


void DataLayer::setup() {
}

void DataLayer::forward() {
}

void DataLayer::backward() {
  
}

Data* DataLayer::get_output() {
  return m_data;
}

void DataLayer::set_current_data(Data* output) {
  m_data = output;
}
