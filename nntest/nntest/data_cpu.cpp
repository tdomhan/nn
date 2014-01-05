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
