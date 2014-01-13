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
#include <vector>

double accuracy(Data* predictions, Data* labels) {
  std::unique_ptr<Data> correct_indicator = std::unique_ptr<Data>(new DataCPU(predictions->get_num_samples(),
                                                           predictions->get_width()));
  std::unique_ptr<Data> labels_matrix = labels->flatten_to_matrix();
  std::unique_ptr<Data> predictions_matrix = predictions->flatten_to_matrix();
  MatrixElementwiseMultiplication(predictions_matrix.get(),
                                  labels_matrix.get(),
                                  correct_indicator.get()).execute();

  /*std::cout << "predictions" << std::endl;
  predictions->print();
  std::cout << "labels" << std::endl;
  labels->print();
  std::cout << "mult" << std::endl;
  correct_indicator->print();*/
  
  double num_correct = DataSum().execute(correct_indicator.get());
  //std::cout << "Num CORRECT: " << num_correct << std::endl;
  double num_total = predictions->get_size_dim(0);
  
  return num_correct / num_total;
}

double accuracy(const std::vector<Data*> &predictions, const std::vector<Data*> &labels) {
  assert(predictions.size() == labels.size());
  int num_correct = 0;
  int num_total = 0;
  std::vector<Data*>::const_iterator current_prediction = predictions.begin();
  std::vector<Data*>::const_iterator current_labels = labels.begin();
  for(;current_prediction != predictions.end();current_prediction++, current_labels++) {
    std::unique_ptr<Data> correct_indicator = std::unique_ptr<Data>(new DataCPU((*current_prediction)->get_num_samples(),
                                                                                (*current_prediction)->get_width()));

    std::unique_ptr<Data> labels_matrix = (*current_labels)->flatten_to_matrix();
    std::unique_ptr<Data> predictions_matrix = (*current_prediction)->flatten_to_matrix();
    
    //std::cout << "labels:" << std::endl;
    //labels_matrix->print();
    //std::cout << "predictions:" << std::endl;
    //predictions_matrix->print();
    MatrixElementwiseMultiplication(predictions_matrix.get(),
                                    labels_matrix.get(),
                                    correct_indicator.get()).execute();
    num_correct += DataSum().execute(correct_indicator.get());
    num_total   += (*current_prediction)->get_num_samples();
  }
  
  return (1.*num_correct) / num_total;
}
