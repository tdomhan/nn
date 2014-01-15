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
  
  DataSet(int batch_size);
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
  
  Data* get_all_train_data() {return m_train_data.get();};
  
  Data* get_all_test_data() {return m_test_data.get();};
  
  int get_data_dimension() {return m_train_data->get_size_dim(1);};
  
  const std::vector<std::unique_ptr<Data>> &get_train_data_batches() {return m_train_data_batches;};
  const std::vector<std::unique_ptr<Data>> &get_train_labels_batches()  {return m_train_labels_batches;};
  
  const std::vector<std::unique_ptr<Data>> &get_test_data_batches() {return m_test_data_batches;};
  const std::vector<std::unique_ptr<Data>> &get_test_labels_batches()  {return m_test_labels_batches;};
  
  bool has_test_data() {return m_has_test_data;};

protected:
  void setup();
  
  void slice_batches(Data& data, std::vector<std::unique_ptr<Data>>& batches);
  
  void load_batch();
  
  std::unique_ptr<Data> m_train_data;
  std::unique_ptr<Data> m_train_labels;
  
  std::vector<std::unique_ptr<Data>> m_train_data_batches;
  std::vector<std::unique_ptr<Data>> m_train_labels_batches;
  
  bool m_has_test_data;
  
  std::unique_ptr<Data> m_test_data;
  std::unique_ptr<Data> m_test_labels;
  
  std::vector<std::unique_ptr<Data>> m_test_data_batches;
  std::vector<std::unique_ptr<Data>> m_test_labels_batches;
  
private:
  int m_batch_size;
  long m_current_pointer;
  long m_rows;

  Data* current_batch_data;
  Data* current_batch_labels;
};

#endif /* defined(__nntest__dataset__) */
