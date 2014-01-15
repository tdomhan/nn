//
//  im2col.h
//  nntest
//
//  Created by Tobias Domhan on 1/14/14.
//  Copyright (c) 2014 Tobias Domhan. All rights reserved.
//

#ifndef __nntest__im2col__
#define __nntest__im2col__

#include <iostream>

#include "data.h"

class Im2Col {
public:
  Im2Col(int stride, int filter_height, int filter_width) :
   m_stride(stride),
   m_filter_height(filter_height),
   m_filter_width(filter_width) {};
  
  //the height of the output matrix
  long get_output_height(Data* matrix_in);
  //the width of the output matrix
  long get_output_width(Data* matrix_in);
  
  //the height of the image after convolving
  long get_height_convolved(Data* matrix_in);
  //the height of the image after convolving
  long get_width_convolved(Data* matrix_in);
  
  //im2col
  void execute(Data* matrix_in, Data* matrix_out);
  
  //col2im
  void reverse(Data* matrix_in, Data* matrix_out);
  
private:
  int m_stride;
  int m_filter_height;
  int m_filter_width;
};

#endif /* defined(__nntest__im2col__) */
