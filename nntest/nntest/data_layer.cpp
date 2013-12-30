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

DataLayer::DataLayer(Data* dataset, int batch_size)
  : m_dataset(dataset),
  m_batch_size(batch_size),
  m_current_pointer(0)
{
  m_rows = dataset->get_size_dim(0);
  m_output = new DataCPU(m_batch_size, m_dataset->get_size_dim(1));
};

void DataLayer::setup() {

}

void DataLayer::forward() {
  Data* batch = m_dataset->get_rows_slice(m_current_pointer, m_current_pointer+m_batch_size);
  m_output->copy_from(*batch);
  delete batch;
}

void DataLayer::backward() {
  
}

Data* DataLayer::get_output() {
  if(batches_remaining()) {
    return m_dataset->get_rows_slice(m_current_pointer, m_current_pointer+m_batch_size);
  } else {
    return NULL;
  }
}

int DataLayer::get_output_size(int dim) {
  return m_dataset->get_size_dim(dim);
}

void DataLayer::next_batch() {
  m_current_pointer += m_batch_size;
}

bool DataLayer::batches_remaining() {
  return m_current_pointer+m_batch_size < m_rows;
}