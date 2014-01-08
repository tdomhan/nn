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
#include <cassert>

MatrixMultiplication::MatrixMultiplication(Data* matrix1, Data* matrix2, Data* result_matrix,
                     MatrixMultiplication::MatrixOp matrix1_transpose,
                     MatrixMultiplication::MatrixOp matrix2_transpose,
                     double alpha,
                     double beta) :
  m_matrix1(matrix1),
  m_matrix2(matrix2),
  m_matrix1_transpose(matrix1_transpose),
  m_matrix2_transpose(matrix2_transpose),
  m_result_matrix(result_matrix),
  m_alpha(alpha),
  m_beta(beta)
{
  check_dimensions();
}

void MatrixMultiplication::check_dimensions() {
  long m1_d0, m1_d1, m2_d0, m2_d1, r_d0, r_d1;
  if(m_matrix1_transpose == NoTranspose) {
    m1_d0 = m_matrix1->get_size_dim(0);
    m1_d1 = m_matrix1->get_size_dim(1);
  } else {
    m1_d0 = m_matrix1->get_size_dim(1);
    m1_d1 = m_matrix1->get_size_dim(0);
  }
  if(m_matrix2_transpose == NoTranspose) {
    m2_d0 = m_matrix2->get_size_dim(0);
    m2_d1 = m_matrix2->get_size_dim(1);
  } else {
    m2_d0 = m_matrix2->get_size_dim(1);
    m2_d1 = m_matrix2->get_size_dim(0);
  }
  r_d0 = m_result_matrix->get_size_dim(0);
  r_d1 = m_result_matrix->get_size_dim(1);

  assert(m1_d1 == m2_d0);
  assert(m1_d0 == r_d0);
  assert(m2_d1 == r_d1);
}

void MatrixMultiplicationMKL::execute() {
  int m1_d0 = (int) m_matrix1->get_size_dim(0);
  int m1_d1 = (int) m_matrix1->get_size_dim(1);
  int m2_d0 = (int) m_matrix2->get_size_dim(0);
  int m2_d1 = (int) m_matrix2->get_size_dim(1);
  
  CBLAS_TRANSPOSE m1_transpose = (m_matrix1_transpose == NoTranspose) ? CblasNoTrans: CblasTrans;
  CBLAS_TRANSPOSE m2_transpose = (m_matrix2_transpose == NoTranspose) ? CblasNoTrans: CblasTrans;
  int m = (m_matrix1_transpose == NoTranspose) ? m1_d0 : m1_d1;
  int n = (m_matrix2_transpose == NoTranspose) ? m2_d1 : m2_d0;
  int k = (m_matrix2_transpose == NoTranspose) ? m2_d0 : m2_d1;
  int lda = (m_matrix1_transpose == NoTranspose) ? k : m;
  int ldb = (m_matrix2_transpose == NoTranspose) ? n : k;
  int ldc = n;
  
  cblas_dgemm(CblasRowMajor,
              m1_transpose,
              m2_transpose,
              m, n, k,
              m_alpha,
              (double*)m_matrix1->get_data(),
              lda,
              (double*)m_matrix2->get_data(),
              ldb,
              m_beta,
              (double*)m_result_matrix->get_data(),
              ldc);
};

void MatrixMultiplicationBasic::execute() {
  //TODO
};


MatrixElementwiseMultiplication::MatrixElementwiseMultiplication(Data* matrix1, Data* matrix2, Data* result_matrix) :
m_matrix1(matrix1),
m_matrix2(matrix2),
m_result_matrix(result_matrix)
{
  check_dimensions();
}

void MatrixElementwiseMultiplication::check_dimensions() {
  long m1_d0 = m_matrix1->get_size_dim(0);
  long m1_d1 = m_matrix1->get_size_dim(1);
  long m2_d0 = m_matrix2->get_size_dim(0);
  long m2_d1 = m_matrix2->get_size_dim(1);
  long r_d0 = m_result_matrix->get_size_dim(0);
  long r_d1 = m_result_matrix->get_size_dim(1);
  
  assert(m1_d0 == m2_d0);
  assert(m2_d0 == r_d0);
  assert(m1_d1 == m2_d1);
  assert(m2_d1 == r_d1);
}

void MatrixElementwiseMultiplication::execute() {
  long dim0 = m_matrix1->get_size_dim(0);
  long dim1 = m_matrix1->get_size_dim(1);
  for (int i = 0; i<dim0; i++) {
    for (int j = 0; j<dim1; j++) {
      double val1 = m_matrix1->get_data_at(i, j);
      double val2 = m_matrix2->get_data_at(i, j);
      m_result_matrix->get_data()[m_result_matrix->get_index(i, j)] = val1*val2;
    }
  }
}

void UniformRandom::execute(Data* matrix) const {
  double* data = matrix->get_data();
  long dim0 = matrix->get_size_dim(0);
  long dim1 = matrix->get_size_dim(1);
  for (int i = 0; i<dim0; i++) {
    for (int j=0; j<dim1; j++) {
      double sign = (rand() % 2) ? -1. : 1.;
      data[matrix->get_index(i, j)] = sign* m_max * rand()/float(RAND_MAX);
      //std::cout << data[matrix->get_index(dim0, dim1)] << std::endl;
    }
  }
}


void SetConst::execute(Data* matrix) const {
  double* data = matrix->get_data();
  long dim0 = matrix->get_size_dim(0);
  long dim1 = matrix->get_size_dim(1);
  for (int i = 0; i<dim0; i++) {
    for (int j=0; j<dim1; j++) {
      data[matrix->get_index(i, j)] = m_value;
      //std::cout << data[matrix->get_index(i, j)] << std::endl;
      //std::cout << i << " " << j << " "<< matrix->get_index(i, j) << std::endl;
    }
  }
}

void MatrixLog::execute(Data* matrix) const {
  double* data = matrix->get_data();
  long dim0 = matrix->get_size_dim(0);
  long dim1 = matrix->get_size_dim(1);
  for (int i = 0; i<dim0; i++) {
    for (int j=0; j<dim1; j++) {
      double value = matrix->get_data_at(i, j);
      data[matrix->get_index(i, j)] = log(fmax(value,0.0001));
    }
  }
}

void AllNegativeZero::execute(Data* matrix) const{
  double* data = matrix->get_data();
  long dim0 = matrix->get_size_dim(0);
  long dim1 = matrix->get_size_dim(1);
  for (int i = 0; i<dim0; i++) {
    for (int j=0; j<dim1; j++) {
      double val = matrix->get_data_at(i, j);
      data[matrix->get_index(i, j)] = fmax(0,val);
    }
  }
}

void AllNegativeZeroMasked::execute(Data* matrix, Data* mask) const{
  double* data = matrix->get_data();
  long dim0 = matrix->get_size_dim(0);
  long dim1 = matrix->get_size_dim(1);
  for (int i = 0; i<dim0; i++) {
    for (int j=0; j<dim1; j++) {
      double val = matrix->get_data_at(i, j);
      double mask_val = mask->get_data_at(i,j);
      data[matrix->get_index(i, j)] = val * (mask_val > 0);
    }
  }
}

void SoftmaxRowByRow::execute(Data* matrix) const {
  long num_rows = matrix->get_size_dim(0);
  long num_columns = matrix->get_size_dim(1);
  for (int row_id=0; row_id < num_rows; row_id++) {
    double row_sum = 0;
    for (int column_id=0; column_id < num_columns; column_id++) {
      row_sum += exp(matrix->get_data_at(row_id, column_id));
    }
    for (int column_id=0; column_id < num_columns; column_id++) {
      double val = matrix->get_data_at(row_id, column_id);
      matrix->get_data()[matrix->get_index(row_id, column_id)] = exp(val) / row_sum;
    }
  }
}

void MatrixAdd::execute(Data* m1, Data* m2) const {
  assert(m1->get_size_dim(0) == m2->get_size_dim(0));
  assert(m1->get_size_dim(1) == m2->get_size_dim(1));
  long num_rows = m1->get_size_dim(0);
  long num_columns = m2->get_size_dim(1);
  for (int row_id=0; row_id < num_rows; row_id++) {
    for (int column_id=0; column_id < num_columns; column_id++) {
      m1->get_data()[m1->get_index(row_id, column_id)] += m_factor * m2->get_data_at(row_id, column_id);
    }
  }
}

void PlusEqualRow::execute(Data* matrix, Data* row) const {
  //make sure row is a vector
  assert(row->get_size_dim(0) == 1);
  //make sure the number of columns matche
  assert(row->get_size_dim(1) == matrix->get_size_dim(1));
  
  long num_rows = matrix->get_size_dim(0);
  long num_columns = matrix->get_size_dim(1);
  for (int row_id=0; row_id < num_rows; row_id++) {
    for (int column_id=0; column_id < num_columns; column_id++) {
      matrix->get_data()[matrix->get_index(row_id, column_id)] += row->get_data_at(0, column_id);
    }
  }
}



double MatrixSum::execute(Data* matrix) {
  double sum = 0;
  long dim0 = matrix->get_size_dim(0);
  long dim1 = matrix->get_size_dim(1);
  for (int i = 0; i<dim0; i++) {
    for (int j=0; j<dim1; j++) {
      double value = matrix->get_data_at(i, j);
      sum += value;
    }
  }
  return sum;
}
