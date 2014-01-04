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

MatrixMultiplication::MatrixMultiplication(Data* matrix1, Data* matrix2, Data* result_matrix) :
m_matrix1(matrix1),
m_matrix2(matrix2),
m_matrix1_transpose(NoTranspose),
m_matrix2_transpose(NoTranspose),
m_result_matrix(result_matrix)
{}

void MatrixMultiplication::check_dimensions() {
  int m1_d0, m1_d1, m2_d0, m2_d1, r_d0, r_d1;
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

MatrixMultiplication::MatrixMultiplication(Data* matrix1, MatrixMultiplication::MatrixOp matrix1_transpose,
                     Data* matrix2, MatrixMultiplication::MatrixOp matrix2_transpose,
                     Data* result_matrix) : MatrixMultiplication(matrix1, matrix2, result_matrix) {
  m_matrix1_transpose = matrix1_transpose;
  m_matrix2_transpose = matrix2_transpose;
}

void MatrixMultiplicationMKL::execute() {
  check_dimensions();

  double alpha = 1.0;
  double beta  = 0.;
  int m1_d0 = m_matrix1->get_size_dim(0);
  int m1_d1 = m_matrix1->get_size_dim(1);
  int m2_d0 = m_matrix2->get_size_dim(0);
  int m2_d1 = m_matrix2->get_size_dim(1);
  
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
              alpha,
              (double*)m_matrix1->get_data(),
              lda,
              (double*)m_matrix2->get_data(),
              ldb,
              beta,
              (double*)m_result_matrix->get_data(),
              ldc);
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

void AllNegativeZero::execute(Data* matrix) const{
  double* data = matrix->get_data();
  int dim0 = matrix->get_size_dim(0);
  int dim1 = matrix->get_size_dim(1);
  for (int i = 0; i<dim0; i++) {
    for (int j=0; j<dim1; j++) {
      double val = matrix->get_data_at(i, j);
      data[matrix->get_index(i, j)] = fmax(0,val);
    }
  }
}

void AllNegativeZeroMasked::execute(Data* matrix, Data* mask) const{
  double* data = matrix->get_data();
  int dim0 = matrix->get_size_dim(0);
  int dim1 = matrix->get_size_dim(1);
  for (int i = 0; i<dim0; i++) {
    for (int j=0; j<dim1; j++) {
      double val = matrix->get_data_at(i, j);
      double mask_val = mask->get_data_at(i,j);
      data[matrix->get_index(i, j)] = val * (mask_val > 0);
    }
  }
}

void SoftmaxRowByRow::execute(Data* matrix) const {
  int num_rows = matrix->get_size_dim(0);
  int num_columns = matrix->get_size_dim(1);
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

