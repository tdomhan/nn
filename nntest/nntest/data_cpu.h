//
//  data_cpu.h
//  nntest
//
//  Created by Tobias Domhan on 12/28/13.
//  Copyright (c) 2013 Tobias Domhan. All rights reserved.
//

#ifndef nntest_data_cpu_h
#define nntest_data_cpu_h

#include "data.h"

class DataCPU : public Data {
public:
  
  DataCPU(int size_dim0, int size_dim1);
  DataCPU(int size_dim0, int size_dim1, double* data);
  ~DataCPU();
  
  virtual Data* get_rows_slice(int start, int end);
  
  virtual double* get_data() const {return m_data; };
  
  virtual void copy_from(const Data& other);
  
  virtual void print();
  
private:
  double* m_data;
  bool m_owns_data;
};

#endif
