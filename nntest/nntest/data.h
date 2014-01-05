//
//  data.h
//  nntest
//
//  Created by Tobias Domhan on 12/28/13.
//  Copyright (c) 2013 Tobias Domhan. All rights reserved.
//

#ifndef nntest_data_h
#define nntest_data_h

#include <cassert>

class Data {
public:
  Data(long size_dim0, long size_dim1) :
    m_size_dim0(size_dim0),
    m_size_dim1(size_dim1) {};
  
  virtual ~Data() {};
  
  long get_index(long idx_dim0, long idx_dim1) const {
    long idx = idx_dim0*m_size_dim1+idx_dim1;
    assert(idx < get_count());
    return idx;
  };
  
  virtual double* get_data() const = 0;
  
  virtual double get_data_at(long row, long column) const {return get_data()[get_index(row, column)]; };
  
  virtual void copy_from(const Data& other) = 0;
  
  //return a data object with a reduced amount of rows
  virtual Data* get_rows_slice(long start, long end) = 0;
  
  long get_count() const {return m_size_dim0*m_size_dim1;};
  
  long get_size_dim(int dim) {
    switch (dim) {
      case 0:
        return m_size_dim0;
        break;
      case 1:
        return m_size_dim1;
      default:
        //TODO: throw exception
        return -1;
        break;
    }
  };
  
  virtual void print() = 0;
  
private:
  //size of the first dimension
  long m_size_dim0;
  //size of the second dimension
  long m_size_dim1;
};


#endif
