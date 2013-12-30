//
//  matrix_multiplication.h
//  nntest
//
//  Created by Tobias Domhan on 12/28/13.
//  Copyright (c) 2013 Tobias Domhan. All rights reserved.
//

#ifndef nntest_matrix_multiplication_h
#define nntest_matrix_multiplication_h

#include "data.h"

class MatrixMultiplication {
public:
  MatrixMultiplication(Data* matrix1, Data* matrix2, Data* result_matrix);
  
  void execute();
  
private:
  Data* m_matrix1;
  Data* m_matrix2;
  Data* m_result_matrix;
};

class UniformRandom {
public:
  UniformRandom(Data* matrix) : m_matrix(matrix) {};
  
  void execute();
private:
  Data* m_matrix;
  
};

class SetConst {
public:
  SetConst(Data* matrix, double value) : m_matrix(matrix), m_value(value) {};
  
  void execute();
private:
  Data* m_matrix;
  double m_value;
};

#endif
