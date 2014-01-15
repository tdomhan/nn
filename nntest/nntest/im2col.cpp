//
//  im2col.cpp
//  nntest
//
//  Created by Tobias Domhan on 1/14/14.
//  Copyright (c) 2014 Tobias Domhan. All rights reserved.
//

#include "im2col.h"


long Im2Col::get_output_height(Data* matrix_in) {
  return matrix_in->get_num_channels() * m_filter_height * m_filter_width;
}

long Im2Col::get_output_width(Data* matrix_in) {
  //the number of patches in the first dimension
  long height_col = get_height_convolved(matrix_in);
  //the number of patches in the second dimension
  long width_col = get_width_convolved(matrix_in);
  return height_col * width_col;
}

long Im2Col::get_height_convolved(Data* matrix_in) {
  return (matrix_in->get_height() - m_filter_height) / m_stride + 1;
}

long Im2Col::get_width_convolved(Data* matrix_in) {
  return (matrix_in->get_width() - m_filter_width) / m_stride + 1;
}

void Im2Col::execute(Data* matrix_in, Data* matrix_out) {
  assert(matrix_in->get_num_samples() == 1);
  assert(matrix_out->get_num_samples() == 1);
  
  long in_height = matrix_in->get_height();
  long in_width = matrix_in->get_width();
  
  //the number of patches in the first dimension
  long height_col = get_height_convolved(matrix_in);
  //the number of patches in the second dimension
  long width_col = get_width_convolved(matrix_in);
  
  long output_rows = get_output_height(matrix_in);
  
  assert(matrix_out->get_height() == output_rows);
  assert(matrix_out->get_width() == height_col * width_col);
  
  double* data_im = matrix_in->get_data();
  double* data_col = matrix_out->get_data();
  
  for (int output_row = 0; output_row < output_rows; ++output_row) {
    int w_offset = output_row % m_filter_width;
    int h_offset = (output_row / m_filter_width) % m_filter_height;
    int c_in = output_row / m_filter_height / m_filter_width;
    for (int h = 0; h < height_col; ++h) {
      for (int w = 0; w < width_col; ++w) {
        data_col[(output_row * height_col + h) * width_col + w] =
        data_im[(c_in * in_height + h * m_stride + h_offset) * in_width
                + w * m_stride + w_offset];
      }
    }
  }
}

void Im2Col::reverse(Data* matrix_in, Data* matrix_out) {
  matrix_out->set_zero();

  assert(matrix_in->get_num_samples() == 1);
  assert(matrix_out->get_num_samples() == 1);
  
  long in_height = matrix_out->get_height();
  long in_width = matrix_out->get_width();
  
  //the number of patches in the first dimension
  long height_col = get_height_convolved(matrix_out);
  //the number of patches in the second dimension
  long width_col = get_width_convolved(matrix_out);
  
  long output_rows = get_output_height(matrix_out);
  
  assert(matrix_in->get_height() == output_rows);
  assert(matrix_in->get_width() == height_col * width_col);
  
  double* data_im = matrix_out->get_data();
  double* data_col = matrix_in->get_data();
  
  for (int output_row = 0; output_row < output_rows; ++output_row) {
    int w_offset = output_row % m_filter_width;
    int h_offset = (output_row / m_filter_width) % m_filter_height;
    int c_in = output_row / m_filter_height / m_filter_width;
    for (int h = 0; h < height_col; ++h) {
      for (int w = 0; w < width_col; ++w) {
        data_im[(c_in * in_height + h * m_stride + h_offset) * in_width + w * m_stride + w_offset] += data_col[(output_row * height_col + h) * width_col + w];
      }
    }
  }
}
