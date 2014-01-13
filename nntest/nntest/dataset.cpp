//
//  dataset.cpp
//  nntest
//
//  Created by Tobias Domhan on 1/5/14.
//  Copyright (c) 2014 Tobias Domhan. All rights reserved.
//

#include "dataset.h"

#include "data_cpu.h"

DataSet::DataSet(int batch_size) : m_batch_size(batch_size), m_has_test_data(false) {};

DataSet::DataSet(Data* data, Data* labels, int batch_size) :
m_batch_size(batch_size),
m_has_test_data(false)
{
    setup();
};

void DataSet::setup() {
  slice_batches(*m_train_data.get(), m_train_data_batches);
  slice_batches(*m_train_labels.get(), m_train_labels_batches);
  if(has_test_data()) {
    slice_batches(*m_test_data.get(), m_test_data_batches);
    slice_batches(*m_test_labels.get(), m_test_labels_batches);
  }
}

void DataSet::slice_batches(Data& data, std::vector<Data*>& batches) {
  int current_row = 0;
  while(current_row < data.get_num_samples()) {
    batches.push_back(data.get_samples_slice(current_row, current_row+m_batch_size));
    current_row += m_batch_size;
  }
}

DataSet::~DataSet() {
}
