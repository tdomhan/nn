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

void MatrixMultiplication::execute() {
  double alpha = 1.0;
  double beta  = 0.;
  int m = m_matrix1->get_size_dim(0);
  int n = m_matrix1->get_size_dim(1);
  int k = m_matrix2->get_size_dim(1);
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


void UniformRandom::execute() {
  double* data = m_matrix->get_data();
  int dim0 = m_matrix->get_size_dim(0);
  int dim1 = m_matrix->get_size_dim(1);
  for (int i = 0; i<dim0; i++) {
    for (int j=0; j<dim1; j++) {
      data[m_matrix->get_index(dim0, dim1)] = rand()/float(RAND_MAX);
      std::cout << data[m_matrix->get_index(dim0, dim1)] << std::endl;
    }
  }
}


void SetConst::execute() {
  double* data = m_matrix->get_data();
  int dim0 = m_matrix->get_size_dim(0);
  int dim1 = m_matrix->get_size_dim(1);
  for (int i = 0; i<dim0; i++) {
    for (int j=0; j<dim1; j++) {
      data[m_matrix->get_index(dim0, dim1)] = m_value;
    }
  }
}

