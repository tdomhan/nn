//
//  matrix_multiplication.cpp
//  nntest
//
//  Created by Tobias Domhan on 12/28/13.
//  Copyright (c) 2013 Tobias Domhan. All rights reserved.
//

#include "math_util.h"

#include "mkl.h"

#include <math.h>

#include <iostream>

MatrixMultiplication::MatrixMultiplication(Data* matrix1, Data* matrix2, Data* result_matrix) :
m_matrix1(matrix1),
m_matrix2(matrix2),
m_result_matrix(result_matrix)
{
  if(m_matrix1->get_size_dim(0) != m_matrix1->get_size_dim(1)) {
    //TODO: throw exception
  }
  if(m_result_matrix->get_size_dim(0) != m_matrix1->get_size_dim(0)){
    
  }
  if(m_result_matrix->get_size_dim(1) != m_matrix1->get_size_dim(1)) {
    
  }
}

void MatrixMultiplicationMKL::execute() {
  double alpha = 1.0;
  double beta  = 0.;
  int m = m_matrix1->get_size_dim(0);
  int k = m_matrix1->get_size_dim(1);
  int n = m_matrix2->get_size_dim(1);
  
  cblas_dgemm(CblasRowMajor,
              CblasNoTrans, CblasNoTrans,
              m, n, k,
              alpha,
              (double*)m_matrix1->get_data(),
              k,
              (double*)m_matrix2->get_data(),
              n,
              beta,
              (double*)m_result_matrix->get_data(),
              n);
};

void MatrixMultiplicationBasic::execute() {
  //TODO
};


void UniformRandom::execute(Data* matrix) const {
  double* data = matrix->get_data();
  int dim0 = matrix->get_size_dim(0);
  int dim1 = matrix->get_size_dim(1);
  for (int i = 0; i<dim0; i++) {
    for (int j=0; j<dim1; j++) {
      data[matrix->get_index(i, j)] = m_max * rand()/float(RAND_MAX);
      //std::cout << data[matrix->get_index(dim0, dim1)] << std::endl;
    }
  }
}


void SetConst::execute(Data* matrix) const {
  double* data = matrix->get_data();
  int dim0 = matrix->get_size_dim(0);
  int dim1 = matrix->get_size_dim(1);
  for (int i = 0; i<dim0; i++) {
    for (int j=0; j<dim1; j++) {
      data[matrix->get_index(i, j)] = m_value;
      //std::cout << data[matrix->get_index(i, j)] << std::endl;
      //std::cout << i << " " << j << " "<< matrix->get_index(i, j) << std::endl;
    }
  }
}


void PlusEqualRow::execute(Data* matrix, Data* row) const {
  //make sure row is a vector
  assert(row->get_size_dim(0) == 1);
  //make sure the number of columns matche
  assert(row->get_size_dim(1) == matrix->get_size_dim(1));
  
  int num_rows = matrix->get_size_dim(0);
  int num_columns = matrix->get_size_dim(1);
  for (int row_id=0; row_id < num_rows; row_id++) {
    for (int column_id=0; column_id < num_columns; column_id++) {
      matrix->get_data()[matrix->get_index(row_id, column_id)] += row->get_data_at(0, column_id);
    }
  }
}

