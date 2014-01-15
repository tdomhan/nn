//
//  tests.h
//  nntest
//
//  Created by Tobias Domhan on 1/15/14.
//  Copyright (c) 2014 Tobias Domhan. All rights reserved.
//

#ifndef __nntest__tests__
#define __nntest__tests__

#include <iostream>
#include <cassert>
#include <cmath>
#include <memory>

#define EPS 0.0001

#include "data.h"
#include "data_cpu.h"
#include "data_layer.h"
#include "linear_layer.h"
#include "relu_layer.h"
#include "conv_layer.h"
#include "softmax_layer.h"
#include "deepnet.h"
#include "math_util.h"


void test_matrix_multiplication();

void test_matrix_multiplication_transpose();

void test_im2col();

//test a layer
void test_layer(Layer* layer, Data* input, Data* expected_output);

void test_layer_gradient(Layer* layer);

void test_linear_layer();

void test_relu_layer();

void test_softmax_layer();

void run_tests();

#endif /* defined(__nntest__tests__) */
