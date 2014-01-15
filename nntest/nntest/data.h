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
#include <memory>
#include <vector>

/**
 * 4D data tensor:
 * num_samples x num_channels x height x width
 *
 * When num_samples == num_channels == 1, it can be used as a height x width matrix
 */
class Data {
public:
  Data(long size_dim2, long size_dim3) :
    m_size_dim0(1),
    m_size_dim1(1),
    m_size_dim2(size_dim2),
    m_size_dim3(size_dim3) {};
  
  Data(long size_dim1, long size_dim2, long size_dim3) :
   m_size_dim0(1),
   m_size_dim1(size_dim1),
   m_size_dim2(size_dim2),
   m_size_dim3(size_dim3) {};
  
  Data(long size_dim0, long size_dim1, long size_dim2, long size_dim3) :
   m_size_dim0(size_dim0),
   m_size_dim1(size_dim1),
   m_size_dim2(size_dim2),
   m_size_dim3(size_dim3) {};
  
  virtual ~Data() {};
  
  virtual std::unique_ptr<Data> copy() = 0;
  
  // flatten all data dimensions, reduces to num_samples x 1 x 1 x data_dimension
  // with data_dimension == num_channels * width * height
  //returns a view of the data
  virtual std::unique_ptr<Data> flatten() = 0;
  
  // flatten all data dimensions, reduces to 1 x 1 x num_samples x data_dimension
  // with data_dimension == num_channels * width * height
  //returns a view of the data
  virtual std::unique_ptr<Data> flatten_to_matrix() = 0;
  
  //Get a view of the Data.
  //Returns a new object, that shares the same data.
  virtual std::unique_ptr<Data> get_view() = 0;
  
  //stack vertically
  // assumes m_size_dim0 == m_size_dim1 == 1
  // and the data is a m_size_dim2 by m_size_dim3 matrix
  virtual std::unique_ptr<Data> vstack(std::vector<Data*> data) = 0;
  
  inline long get_index(long idx_dim3) const {
    long idx = idx_dim3;
    assert(idx < get_total_count());
    return idx;
  };
  
  //get the matrix index
  inline long get_index(long idx_dim2, long idx_dim3) const {
    assert(m_size_dim0 == 1);
    assert(m_size_dim1 == 1);
    long idx = idx_dim2*m_size_dim3+idx_dim3;
    assert(idx < get_total_count());
    return idx;
  };
  
  inline long get_index(long idx_dim1, long idx_dim2, long idx_dim3) const {
    long idx = idx_dim1*(m_size_dim2*m_size_dim3)+idx_dim2*m_size_dim3+idx_dim3;
    assert(idx < get_total_count());
    return idx;
  };
  
  inline long get_index(long idx_dim0, long idx_dim1, long idx_dim2, long idx_dim3) const {
    long idx = idx_dim0*(m_size_dim1*m_size_dim2*m_size_dim3)+idx_dim1*(m_size_dim2*m_size_dim3)+idx_dim2*m_size_dim3+idx_dim3;
    assert(idx < get_total_count());
    return idx;
  };
  
  virtual double* get_data() const = 0;
  
  inline double get_data_at(long column) const {return get_data()[get_index(column)]; };
  
  inline double get_data_at(long row, long column) const {return get_data()[get_index(row, column)]; };
  
  inline double get_data_at(long channel, long row, long column) const {return get_data()[get_index(channel, row, column)]; };
  
  inline double get_data_at(long sample_id, long channel, long row, long column) const {return get_data()[get_index(sample_id, channel, row, column)]; };
  
  virtual void copy_from(const Data& other) = 0;
  
  //return a data object with a reduced amount of rows
  //start: inclusive
  //end: exclusive
  virtual std::unique_ptr<Data> get_rows_slice(long dim2_start, long dim2_end) = 0;
  
  //start: inclusive
  //end: exclusive
  virtual std::unique_ptr<Data> get_samples_slice(long dim0_start, long dim0_end) = 0;
  
  inline long get_total_count() const {return m_size_dim0*m_size_dim1*m_size_dim2*m_size_dim3;};
  
  inline long get_count_per_sample() const {return m_size_dim1*m_size_dim2*m_size_dim3;};
  
  inline long get_size_dim(int dim) {
    switch (dim) {
      case 0:
        return m_size_dim0;
        break;
      case 1:
        return m_size_dim1;
      case 2:
        return m_size_dim2;
      case 3:
        return m_size_dim3;
      default:
        //TODO: throw exception
        return -1;
        break;
    }
  };
  
  inline void reshape(long size_dim0, long size_dim1, long size_dim2, long size_dim3) {
    long old_count = get_total_count();
    m_size_dim0 = size_dim0;
    m_size_dim1 = size_dim1;
    m_size_dim2 = size_dim2;
    m_size_dim3 = size_dim3;
    long new_count = get_total_count();
    assert(old_count == new_count);
  };
  
  inline long get_num_samples() {return m_size_dim0;};
  inline long get_num_channels() {return m_size_dim1;};
  inline long get_height() {return m_size_dim2;};
  inline long get_width() {return m_size_dim3;};
  
  inline int get_effective_dimension() {
    int dim = 0;
    if (m_size_dim0 > 1)
      dim += 1;
    if (m_size_dim1 > 1)
      dim += 1;
    if (m_size_dim2 > 1)
      dim += 1;
    if (m_size_dim3 > 1)
      dim += 1;
    return dim;
  };
  
  virtual void set_zero() = 0;
  
  virtual void print() = 0;
  
private:
  //size of the first dimension (number of samples)
  long m_size_dim0;
  //size of the second dimension
  long m_size_dim1;
  long m_size_dim2;
  long m_size_dim3;
};


#endif
