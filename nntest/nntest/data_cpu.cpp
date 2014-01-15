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

DataCPU::DataCPU(long size_dim2, long size_dim3)
  : Data(size_dim2, size_dim3), m_owns_data(true) {
    allocate();
}

DataCPU::DataCPU(long size_dim1, long size_dim2, long size_dim3)
: Data(size_dim1, size_dim2, size_dim3), m_owns_data(true) {
  allocate();
}


DataCPU::DataCPU(long size_dim0, long size_dim1, long size_dim2, long size_dim3)
 : Data(size_dim0, size_dim1, size_dim2, size_dim3), m_owns_data(true){
   allocate();
}

DataCPU::DataCPU(long size_dim2, long size_dim3, double* data) :
Data(size_dim2, size_dim3),
m_data(data),
m_owns_data(false)
{
}

DataCPU::DataCPU(long size_dim1, long size_dim2, long size_dim3, double* data) :
Data(size_dim1, size_dim2, size_dim3),
m_data(data),
m_owns_data(false) {
  
}


DataCPU::DataCPU(long size_dim0, long size_dim1, long size_dim2, long size_dim3, double* data) :
Data(size_dim0, size_dim1, size_dim2, size_dim3),
m_data(data),
m_owns_data(false) {
  
}

DataCPU::~DataCPU() {
  if (m_owns_data) {
    //std::cout << "Freeing DataCPU(" << get_num_samples() << ", " << get_num_channels() << ", " << get_height() << ", " << get_width() << ", "  << ")" << std::endl;
    mkl_free(m_data);
  }
}

std::unique_ptr<Data> DataCPU::copy() {
  Data* new_object = new DataCPU(get_size_dim(0), get_size_dim(1), get_size_dim(2), get_size_dim(3));
  new_object->copy_from(*this);
  return std::unique_ptr<Data>(new_object);
}

std::unique_ptr<Data> DataCPU::flatten() {
  Data* new_object = new DataCPU(get_num_samples(), 1, 1, get_count_per_sample(), m_data);
  return std::unique_ptr<Data>(new_object);
}

std::unique_ptr<Data> DataCPU::flatten_to_matrix() {
  Data* new_object = new DataCPU(1, 1, get_num_samples(), get_count_per_sample(), m_data);
  return std::unique_ptr<Data>(new_object);
}

std::unique_ptr<Data> DataCPU::get_view() {
  Data* new_object = new DataCPU(get_size_dim(0),
                                 get_size_dim(1),
                                 get_size_dim(2),
                                 get_size_dim(3),
                                 m_data);
  return std::unique_ptr<Data>(new_object);
}

std::unique_ptr<Data> DataCPU::vstack(std::vector<Data*> data) {
  assert(data.size() > 0);
  long dim2 = 0;
  long dim3 = data[0]->get_size_dim(3);
  for (std::vector<Data*>::iterator it=data.begin(); it<data.end(); it++) {
    dim2 += (*it)->get_size_dim(2);
    assert(dim3 == (*it)->get_size_dim(3));
  }
  Data* stacked = new DataCPU(dim2,dim3);
  int stacked_row = 0;
  for (std::vector<Data*>::iterator it=data.begin(); it<data.end(); it++) {
    Data* current_data = *it;
    for (int current_row=0; current_row<current_data->get_size_dim(2); current_row++) {
      for (int column=0; column<dim3; column++) {
        double val = current_data->get_data_at(current_row, column);
        stacked->get_data()[stacked->get_index(stacked_row, column)] = val;
      }
      stacked_row++;
    }
  }
  return std::unique_ptr<Data>(stacked);
}

std::unique_ptr<Data> DataCPU::get_rows_slice(long start, long end) {
  long dim2 = end-start;
  Data* ret = new DataCPU(dim2,
                          get_size_dim(3),
                          &m_data[start*get_size_dim(3)]);
  return std::unique_ptr<Data>(ret);
}

std::unique_ptr<Data> DataCPU::get_samples_slice(long dim0_start, long dim0_end) {
  assert(dim0_end > dim0_start);
  long dim0 = dim0_end-dim0_start;
  long count_per_sample = get_count_per_sample();
  Data* ret = new DataCPU(dim0,
                          get_size_dim(1),
                          get_size_dim(2),
                          get_size_dim(3),
                          &m_data[dim0_start*count_per_sample]);
  return std::unique_ptr<Data>(ret);
}

void DataCPU::copy_from(const Data& other) {
  assert(get_total_count()<=other.get_total_count());
  double* other_data = other.get_data();
  memcpy(m_data, other_data, get_total_count()*sizeof( double ));
}

void DataCPU::set_zero() {
  memset(m_data, 0, get_total_count()*sizeof( double ));
}

void DataCPU::print() {
  for(int i=0;i<get_height();i++) {
    for(int j=0; j<get_width(); j++) {
      std::cout << get_data()[get_index(i, j)] << " ";
    }
    std::cout << std::endl;
  }
}

void DataCPU::allocate() {
  m_data = (double *)mkl_malloc( get_total_count()*sizeof( double ), 64 );
}
