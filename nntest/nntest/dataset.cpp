//
//  dataset.cpp
//  nntest
//
//  Created by Tobias Domhan on 1/5/14.
//  Copyright (c) 2014 Tobias Domhan. All rights reserved.
//

#include "dataset.h"

#include "data_cpu.h"

DataSet::DataSet(int batch_size) : m_batch_size(batch_size), m_current_pointer(0) {};

DataSet::DataSet(Data* data, Data* labels, int batch_size) :
m_batch_size(batch_size),
m_current_pointer(0),
m_data(data),
m_labels(labels)
{
    setup();
};

void DataSet::setup() {
  assert(m_data->get_size_dim(0) == m_labels->get_size_dim(0));
  m_rows = m_data->get_size_dim(0);
  current_batch_data = new DataCPU(m_batch_size, m_data->get_size_dim(1));
  current_batch_labels = new DataCPU(m_batch_size, m_labels->get_size_dim(1));
  
  load_batch();
}

DataSet::~DataSet() {
  delete m_data;
  delete m_labels;
  delete current_batch_data;
  delete current_batch_labels;
}

void DataSet::next_batch() {
  assert(batches_remaining());
  m_current_pointer += m_batch_size;
  
  load_batch();
}

bool DataSet::batches_remaining() {
  return m_current_pointer+m_batch_size < m_rows;
}


void DataSet::reset() {
  m_current_pointer = 0;

  load_batch();
}

void DataSet::load_batch() {
  Data* batch;
  batch= m_data->get_rows_slice(m_current_pointer, m_current_pointer+m_batch_size);
  current_batch_data->copy_from(*batch);
  delete batch;
  
  batch= m_labels->get_rows_slice(m_current_pointer, m_current_pointer+m_batch_size);
  current_batch_labels->copy_from(*batch);
  delete batch;
}