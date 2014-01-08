//
//  metrics.cpp
//  nntest
//
//  Created by Tobias Domhan on 1/7/14.
//  Copyright (c) 2014 Tobias Domhan. All rights reserved.
//

#include "metrics.h"

#include "math_util.h"
#include "data_cpu.h"

#include <memory>

double accuracy(Data* predictions, Data* labels) {
  std::unique_ptr<Data> correct_indicator = std::unique_ptr<Data>(new DataCPU(predictions->get_size_dim(0),
                                                           predictions->get_size_dim(1)));
  MatrixElementwiseMultiplication(predictions, labels, correct_indicator.get()).execute();

  /*std::cout << "predictions" << std::endl;
  predictions->print();
  std::cout << "labels" << std::endl;
  labels->print();
  std::cout << "mult" << std::endl;
  correct_indicator->print();*/
  
  double num_correct = MatrixSum().execute(correct_indicator.get());
  //std::cout << "Num CORRECT: " << num_correct << std::endl;
  double num_total = predictions->get_size_dim(0);
  
  return num_correct / num_total;
}
