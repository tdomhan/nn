//
//  net.cpp
//  nntest
//
//  Created by Tobias Domhan on 1/1/14.
//  Copyright (c) 2014 Tobias Domhan. All rights reserved.
//

#include "deepnet.h"

#include "data_layer.h"

#include <exception>

using namespace std;

DeepNetwork::DeepNetwork(int batch_size, int data_dimension) : m_setup(false) {
  m_layers.push_back(std::unique_ptr<Layer>(new DataLayer(batch_size, data_dimension)));
};

std::unique_ptr<Data> DeepNetwork::get_output() {
  return m_layers.back()->get_output()->copy();
}

void DeepNetwork::setup() {
  vector<unique_ptr<Layer>>::iterator it = m_layers.begin();
  for (; it<m_layers.end(); it++) {
    (*it)->setup();
  }
  m_setup = true;
}

void DeepNetwork::forward(Data* input_data) {
  check_setup();
  
  dynamic_cast<DataLayer*>(m_layers[0].get())->set_current_data(input_data);

  vector<unique_ptr<Layer>>::iterator it = m_layers.begin();
  for (; it<m_layers.end(); it++) {
    (*it)->forward();
  }
}

void DeepNetwork::backward(Data* expected_output) {
  check_setup();
  
  dynamic_cast<LossLayer*>(m_layers.back().get())->backward(expected_output);
  
  vector<unique_ptr<Layer>>::reverse_iterator it = m_layers.rbegin();
  //we already covered the last layer, so skip it now:
  it++;
  //the rest of the layers:
  for (; it<m_layers.rend(); it++) {
    (*it)->backward();
  }
}

void DeepNetwork::update(double learning_rate) {
  vector<unique_ptr<Layer>>::iterator it = m_layers.begin();
  for (; it<m_layers.end(); it++) {
    (*it)->update(learning_rate);
  }
}

void DeepNetwork::add_layer(std::unique_ptr<Layer> layer) {
  m_layers.back()->connect_top(layer.get());
  layer->connect_bottom(m_layers.back().get());

  m_layers.push_back(std::move(layer));
}

void DeepNetwork::check_setup() {
  if(!m_setup) {
    throw runtime_error("setup() was not called before using the network");
  }
}
