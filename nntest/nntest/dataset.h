//
//  dataset.h
//  nntest
//
//  Created by Tobias Domhan on 1/5/14.
//  Copyright (c) 2014 Tobias Domhan. All rights reserved.
//

#ifndef __nntest__dataset__
#define __nntest__dataset__

#include <iostream>
#include <vector>

#include "data.h"

class DataSet {
public:
  
  DataSet(int batch_size) : m_batch_size(batch_size) {};
  /**
   data: m x n matrix, the input data
   label: m x l matrix, one-hot encoded labels

   with
   l: num labels
   m: num_samples
   n: input dimension
   */
  DataSet(Data* data, Data* labels, int batch_size);
  
  ~DataSet();
  
  Data* get_batch_data();
  
  Data* get_batch_labels();
  
  void next_batch();
  
  bool batches_remaining();

protected:
  Data* m_data;
  Data* m_labels;
  
private:
  int m_batch_size;
  long m_current_pointer;
  long m_rows;

  Data* current_batch_data;
  Data* current_batch_labels;
};

#endif /* defined(__nntest__dataset__) */
