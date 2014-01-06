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


DataLayer::DataLayer(Data* output)
  : m_output(output)
{
};

void DataLayer::setup() {
}

void DataLayer::forward() {
}

void DataLayer::backward() {
  
}

Data* DataLayer::get_output() {
  return m_output;
}

void DataLayer::set_current_output(Data* output) {
  m_output = output;
}
