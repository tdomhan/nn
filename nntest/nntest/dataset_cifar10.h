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

class DataSetCIFAR10 : public DataSet {
public:
  DataSetCIFAR10(const char* fname, int batch_size);
  
private:
  void read_file(const char* fname);
};

#endif /* defined(__nntest__dataset_cifar10__) */
