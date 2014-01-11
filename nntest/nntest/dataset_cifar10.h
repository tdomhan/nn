//
//  dataset_cifar10.h
//  nntest
//
//  Created by Tobias Domhan on 1/5/14.
//  Copyright (c) 2014 Tobias Domhan. All rights reserved.
//

#ifndef __nntest__dataset_cifar10__
#define __nntest__dataset_cifar10__

#include <iostream>

#include "dataset.h"

#include <string>
#include <memory>

class DataSetCIFAR10 : public DataSet {
public:
  /*
   data_folder: the folder the dataset is in(without a trailing slash). In this folder we expecte the following folder structure:
                cifar-10-batches-bin/data_batch_*.bin
   fold: the fold that is left of as the test set. (0 is the official CIFAR-10 test set.)
   batch_size: the size of the individual batches
   */
  DataSetCIFAR10(const std::string &data_folder, int fold, int batch_size);
  
private:
};


class DataSetCIFAR10File {
public:
  DataSetCIFAR10File(const std::string &fname) {read_file(fname);};
  
  Data* get_data() {return m_data.get();};
  Data* get_labels() {return m_labels.get();};
  
  std::unique_ptr<Data> get_data_permenantly() {return std::move(m_data);};
  std::unique_ptr<Data> get_labels_permenantly() {return std::move(m_labels);};
private:
  /*
   Read the data file.
   */
  void read_file(const std::string &fname);

  std::unique_ptr<Data> m_data;
  std::unique_ptr<Data> m_labels;
};

#endif /* defined(__nntest__dataset_cifar10__) */
