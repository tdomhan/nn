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
  
  DataCPU(long size_dim2, long size_dim3);
  DataCPU(long size_dim1, long size_dim2, long size_dim3);
  DataCPU(long size_dim0, long size_dim1, long size_dim2, long size_dim3);

  DataCPU(long size_dim2, long size_dim3, double* data);
  DataCPU(long size_dim1, long size_dim2, long size_dim3, double* data);
  DataCPU(long size_dim0, long size_dim1, long size_dim2, long size_dim3, double* data);
  
  
  virtual ~DataCPU();

  virtual std::unique_ptr<Data> copy();
  
  virtual std::unique_ptr<Data> flatten();
  
  virtual std::unique_ptr<Data> flatten_to_matrix();
  
  virtual std::unique_ptr<Data> get_view();
  
  virtual std::unique_ptr<Data> vstack(std::vector<Data*> data);
  
  virtual Data* get_rows_slice(long dim2_start, long dim2_end);
  
  virtual Data* get_samples_slice(long dim0_start, long dim0_end);
  
  virtual double* get_data() const {return m_data; };
  
  virtual void copy_from(const Data& other);
  
  virtual void print(); 
  
private:
  void allocate();
  
  double* m_data;
  bool m_owns_data;
};

#endif
