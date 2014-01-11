//
//  data_cpu.cpp
//  nntest
//
//  Created by Tobias Domhan on 12/28/13.
//  Copyright (c) 2013 Tobias Domhan. All rights reserved.
//

#include "data_cpu.h"

#include "mkl.h"

#include <stdio.h>
#include <string.h>
#include <cassert>
#include <iostream>


DataCPU::DataCPU(long size_dim0, long size_dim1, double* data) :
  Data(size_dim0, size_dim1),
  m_data(data),
  m_owns_data(false)
{
  
}

DataCPU::DataCPU(long size_dim0, long size_dim1)
  : Data(size_dim0, size_dim1), m_owns_data(true) {
    m_data = (double *)mkl_malloc( get_count()*sizeof( double ), 64 );
}

DataCPU::~DataCPU() {
  if (m_owns_data) {
    mkl_free(m_data);
  }
}

std::unique_ptr<Data> DataCPU::copy() {
  Data* new_object = new DataCPU(get_size_dim(0), get_size_dim(1));
  new_object->copy_from(*this);
  return std::unique_ptr<Data>(new_object);
}

std::unique_ptr<Data> DataCPU::vstack(std::vector<Data*> data) {
  assert(data.size() > 0);
  long dim0 = 0;
  long dim1 = data[0]->get_size_dim(1);
  for (std::vector<Data*>::iterator it=data.begin(); it<data.end(); it++) {
    dim0 += (*it)->get_size_dim(0);
    assert(dim1 == (*it)->get_size_dim(1));
  }
  Data* stacked = new DataCPU(dim0,dim1);
  int stacked_row = 0;
  for (std::vector<Data*>::iterator it=data.begin(); it<data.end(); it++) {
    Data* current_data = *it;
    for (int current_row=0; current_row<current_data->get_size_dim(0); current_row++) {
      for (int column=0; column<dim1; column++) {
        double val = current_data->get_data_at(current_row, column);
        stacked->get_data()[stacked->get_index(stacked_row, column)] = val;
      }
      stacked_row++;
    }
  }
  return std::unique_ptr<Data>(stacked);
}

Data* DataCPU::get_rows_slice(long start, long end) {
  long dim0 = end-start;
  Data* ret = new DataCPU(dim0, get_size_dim(1), &m_data[start*get_size_dim(1)]);
  return ret;
}

void DataCPU::copy_from(const Data& other) {
  assert(get_count()<=other.get_count());
  double* other_data = other.get_data();
  memcpy(m_data, other_data, get_count()*sizeof( double ));
}

void DataCPU::print() {
  for(int i=0;i<get_size_dim(0);i++) {
    for(int j=0; j<get_size_dim(1); j++) {
      std::cout << get_data()[i*get_size_dim(1)+j] << " ";
    }
    std::cout << std::endl;
  }
}
