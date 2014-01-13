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
  assert(m_matrix1->get_num_samples() == 1);
  assert(m_matrix2->get_num_samples() == 1);
  assert(m_result_matrix->get_num_samples() == 1);

  assert(m_matrix1->get_num_channels() == 1);
  assert(m_matrix2->get_num_channels() == 1);
  assert(m_result_matrix->get_num_channels() == 1);
  
  long m1_d0, m1_d1, m2_d0, m2_d1, r_d0, r_d1;
  if(m_matrix1_transpose == NoTranspose) {
    m1_d0 = m_matrix1->get_height();
    m1_d1 = m_matrix1->get_width();
  } else {
    m1_d0 = m_matrix1->get_width();
    m1_d1 = m_matrix1->get_height();
  }
  if(m_matrix2_transpose == NoTranspose) {
    m2_d0 = m_matrix2->get_height();
    m2_d1 = m_matrix2->get_width();
  } else {
    m2_d0 = m_matrix2->get_width();
    m2_d1 = m_matrix2->get_height();
  }
  r_d0 = m_result_matrix->get_height();
  r_d1 = m_result_matrix->get_width();

  assert(m1_d1 == m2_d0);
  assert(m1_d0 == r_d0);
  assert(m2_d1 == r_d1);
}

void MatrixMultiplicationMKL::execute() {
  int m1_d0 = (int) m_matrix1->get_height();
  int m1_d1 = (int) m_matrix1->get_width();
  int m2_d0 = (int) m_matrix2->get_height();
  int m2_d1 = (int) m_matrix2->get_width();
  
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
  assert(m_matrix1->get_num_samples() == 1);
  assert(m_matrix2->get_num_samples() == 1);
  assert(m_result_matrix->get_num_samples() == 1);
  
  assert(m_matrix1->get_num_channels() == 1);
  assert(m_matrix2->get_num_channels() == 1);
  assert(m_result_matrix->get_num_channels() == 1);
  
  long m1_d0 = m_matrix1->get_height();
  long m1_d1 = m_matrix1->get_width();
  long m2_d0 = m_matrix2->get_height();
  long m2_d1 = m_matrix2->get_width();
  long r_d0 = m_result_matrix->get_height();
  long r_d1 = m_result_matrix->get_width();
  
  assert(m1_d0 == m2_d0);
  assert(m2_d0 == r_d0);
  assert(m1_d1 == m2_d1);
  assert(m2_d1 == r_d1);
}

void MatrixElementwiseMultiplication::execute() {
  long dim0 = m_matrix1->get_height();
  long dim1 = m_matrix1->get_width();

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
  long count = matrix->get_total_count();
  for (int i = 0; i<count; i++) {
    double sign = (rand() % 2) ? -1. : 1.;
    data[i] = sign* m_max * rand()/float(RAND_MAX);
    //std::cout << data[matrix->get_index(dim0, dim1)] << std::endl;
  }
}


void SetConst::execute(Data* matrix) const {
  double* data = matrix->get_data();
  long count = matrix->get_total_count();
  for (int i = 0; i<count; i++) {
      data[i] = m_value;
      //std::cout << data[matrix->get_index(i, j)] << std::endl;
      //std::cout << i << " " << j << " "<< matrix->get_index(i, j) << std::endl;
  }
}

void MatrixLog::execute(Data* matrix) const {
  double* data = matrix->get_data();
  long count = matrix->get_total_count();
  for (int i = 0; i<count; i++) {
    double value = data[i];
    data[i] = log(fmax(value,0.0001));
  }
}

void AllNegativeZero::execute(Data* matrix) const{
  double* data = matrix->get_data();
  long count = matrix->get_total_count();
  for (int i = 0; i<count; i++) {
    double val = data[i];
    data[i] = fmax(0,val);
  }
}

void AllNegativeZeroMasked::execute(Data* matrix, Data* mask) const{
  double* data = matrix->get_data();
  double* mask_data = mask->get_data();
  assert(matrix->get_total_count() == mask->get_total_count());
  long count = matrix->get_total_count();
  for (int i = 0; i<count; i++) {
    double val = data[i];
    double mask_val = mask_data[i];
    data[i] = val * (mask_val > 0);
  }
}

void SoftmaxRowByRow::execute(Data* matrix) const {
  assert(matrix->get_num_channels() == 1);
  assert(matrix->get_height() == 1);

  long num_rows = matrix->get_num_samples();
  long num_columns = matrix->get_width();
  for (int row_id=0; row_id < num_rows; row_id++) {
    double row_sum = 0;
    for (int column_id=0; column_id < num_columns; column_id++) {
      row_sum += exp(matrix->get_data_at(row_id, 0, 0, column_id));
    }
    for (int column_id=0; column_id < num_columns; column_id++) {
      double val = matrix->get_data_at(row_id, 0, 0, column_id);
      matrix->get_data()[matrix->get_index(row_id,  0, 0, column_id)] = exp(val) / row_sum;
    }
  }
}

void MatrixAdd::execute(Data* m1, Data* m2) const {
  assert(m1->get_num_samples() == 1);
  assert(m1->get_num_samples() == 1);
  assert(m2->get_num_samples() == 1);
  assert(m2->get_num_samples() == 1);
  
  assert(m1->get_height() == m2->get_height());
  assert(m1->get_width() == m2->get_width());
  long num_rows = m1->get_height();
  long num_columns = m2->get_width();
  for (int row_id=0; row_id < num_rows; row_id++) {
    for (int column_id=0; column_id < num_columns; column_id++) {
      m1->get_data()[m1->get_index(row_id, column_id)] += m_factor * m2->get_data_at(row_id, column_id);
    }
  }
}

void PlusEqualRow::execute(Data* matrix, Data* row) const {
  //make sure row is a vector
  assert(row->get_height() == 1);
  //make sure the number of columns matche
  assert(row->get_width() == matrix->get_width());
  
  long num_rows = matrix->get_height();
  long num_columns = matrix->get_width();
  for (int row_id=0; row_id < num_rows; row_id++) {
    for (int column_id=0; column_id < num_columns; column_id++) {
      matrix->get_data()[matrix->get_index(row_id, column_id)] += row->get_data_at(0, column_id);
    }
  }
}



double DataSum::execute(Data* matrix) {
  double* data = matrix->get_data();
  double sum = 0;
  long count = matrix->get_total_count();
  for (int i = 0; i<count; i++) {
      sum += data[i];
  }
  return sum;
}

std::unique_ptr<Data> MaxProbabilityPrediction::execute(Data* probabilities) {
  assert(probabilities->get_num_channels() == 1);
  assert(probabilities->get_height() == 1);
  
  std::unique_ptr<Data> predictions(probabilities->copy());
  
  SetConst(0).execute(predictions.get());
  
  long num_samples = probabilities->get_num_samples();
  long num_columns = probabilities->get_width();
  for (long sample=0; sample<num_samples; sample++) {
    double max = -1.;
    long max_col = -1;
    for (long column=0; column<num_columns; column++) {
      double val = probabilities->get_data_at(sample, 0, 0, column);
      if (val > max) {
        max = val;
        max_col = column;
      }
    }
    predictions->get_data()[predictions->get_index(sample, 0, 0, max_col)] = 1;
  }
  
  return predictions;
}
