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
  Data(int size_dim0, int size_dim1) :
    m_size_dim0(size_dim0),
    m_size_dim1(size_dim1) {};
  
  virtual ~Data() {};
  
  int get_index(int idx_dim0, int idx_dim1) const {
    int idx = idx_dim0*m_size_dim1+idx_dim1;
    assert(idx < get_count());
    return idx;
  };
  
  virtual double* get_data() const = 0;
  
  virtual double get_data_at(int row, int column) const {return get_data()[get_index(row, column)]; };
  
  virtual void copy_from(const Data& other) = 0;
  
  //return a data object with a reduced amount of rows
  virtual Data* get_rows_slice(int start, int end) = 0;
  
  int get_count() const {return m_size_dim0*m_size_dim1;};
  
  int get_size_dim(int dim) {
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
  int m_size_dim0;
  //size of the second dimension
  int m_size_dim1;
};


#endif
