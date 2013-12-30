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
  
  virtual void execute() = 0;
  
protected:
  Data* m_matrix1;
  Data* m_matrix2;
  Data* m_result_matrix;
};

class MatrixMultiplicationBasic : public MatrixMultiplication {
public:
  MatrixMultiplicationBasic(Data* matrix1, Data* matrix2, Data* result_matrix)
    : MatrixMultiplication(matrix1, matrix2, result_matrix) {};
  
  virtual void execute();
private:
  
};

class MatrixMultiplicationMKL :  public MatrixMultiplication {
public:
  MatrixMultiplicationMKL(Data* matrix1, Data* matrix2, Data* result_matrix)
    : MatrixMultiplication(matrix1, matrix2, result_matrix) {};
  virtual void execute();
private:
};

class UnaryMathOp {
public:
  virtual void execute(Data* matrix) const = 0;
  virtual ~UnaryMathOp() {};
private:
};

class UniformRandom : public UnaryMathOp {
public:
  UniformRandom(double max) : m_max(max) {};
  
  virtual void execute(Data* matrix) const;
private:
  double m_max;
};

class SetConst : public UnaryMathOp {
public:
  SetConst(double value) : m_value(value) {};
  
  virtual void execute(Data* matrix) const;
private:
  double m_value;
};

class DualMathOp {
public:
  virtual void execute(Data* matrix1, Data* matrix2) const = 0;
  virtual ~DualMathOp() {};
private:
};

/**
 * Adds a vector row by row
 */
class PlusEqualRow: public DualMathOp {
public:
  virtual void execute(Data* matrix, Data* row) const;
  virtual ~PlusEqualRow() {};
private:
};

#endif
