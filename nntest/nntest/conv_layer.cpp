//
//  conv_layer.cpp
//  nntest
//
//  Created by Tobias Domhan on 1/13/14.
//  Copyright (c) 2014 Tobias Domhan. All rights reserved.
//

#include "conv_layer.h"

#include "data_cpu.h"

#include <iostream>


ConvLayer::ConvLayer(int num_filters, int stride, int filter_height, int filter_width)
: m_num_filters(num_filters),
  m_stride(stride),
  m_filter_height(filter_height),
  m_filter_width(filter_width),
  im2col(stride, filter_height, filter_width),
  m_initialize(new UniformRandom(0.05))
{
  
}

ConvLayer::ConvLayer(int num_filters,
                     int stride,
                     int filter_height,
                     int filter_width,
                     UnaryMathOp const* initialize)
: m_num_filters(num_filters),
  m_stride(stride),
  m_filter_height(filter_height),
  m_filter_width(filter_width),
  im2col(stride, filter_height, filter_width),
  m_initialize(initialize)
{
  
}


ConvLayer::~ConvLayer() {
  delete m_initialize;
}

void ConvLayer::setup() {
  assert(has_bottom_layer());
  
  initialize_weights();
  
  initialize_bias();
  
  Data* bottom_out = get_bottom_layer()->get_output();
  
  //for now only handle a single example at a time
  //assert(get_bottom_layer()->get_output()->get_num_samples() == 1);
  
  m_output = std::unique_ptr<Data>(new DataCPU(get_bottom_layer()->get_output()->get_num_samples(),
                                               m_num_filters,
                                               im2col.get_height_convolved(bottom_out),
                                               im2col.get_width_convolved(bottom_out)));

/*  m_output_matrix = m_output->get_view();
  m_output_matrix->reshape(1,
                           1,
                           m_num_filters,
                           im2col.get_output_width(bottom_out));*/
  
  m_col_image = std::unique_ptr<Data>(new DataCPU(
                                                  1,
                                                  1,
                                                  im2col.get_output_height(bottom_out),
                                                  im2col.get_output_width(bottom_out)));
  
  //provide the error to the layer below
  m_backprop_error = std::unique_ptr<Data>(new DataCPU(bottom_out->get_size_dim(0),
                                                       bottom_out->get_size_dim(1),
                                                       bottom_out->get_size_dim(2),
                                                       bottom_out->get_size_dim(3)));
}

void ConvLayer::initialize_weights() {
  Data* bottom_out = get_bottom_layer()->get_output();
  m_weights = std::unique_ptr<Data>(new DataCPU(1,
                                                1,
                                                m_num_filters,
                                                im2col.get_output_height(bottom_out)));
  m_weights_update = std::unique_ptr<Data>(new DataCPU(1,
                                                       1,
                                                       m_weights->get_height(),
                                                       m_weights->get_width()));
  m_initialize->execute(m_weights.get());
}

void ConvLayer::initialize_bias() {
  m_bias = std::unique_ptr<Data>(new DataCPU(1, m_num_filters));
  m_bias_update = std::unique_ptr<Data>(new DataCPU(1, m_num_filters));
  m_initialize->execute(m_bias.get());
}


void ConvLayer::forward() {
  Data* bottom_out = get_bottom_layer()->get_output();
  
  for(int sample=0; sample < bottom_out->get_num_samples(); sample++) {
    std::unique_ptr<Data> bottom_out_sample = bottom_out->get_samples_slice(sample, sample+1);
    std::unique_ptr<Data> sample_out_matrix = m_output->get_samples_slice(sample, sample+1);
    sample_out_matrix->reshape(1,
                               1,
                               m_num_filters,
                               im2col.get_output_width(bottom_out));

    im2col.execute(bottom_out_sample.get(), m_col_image.get());
    MatrixMultiplicationMKL(m_weights.get(),
                            m_col_image.get(),
                            sample_out_matrix.get()).execute();
  }
}

void ConvLayer::backward() {
  Data* top_error = get_top_layer()->get_backprop_error();
  Data* bottom_out = get_bottom_layer()->get_output();
  
  //assert(top_error->get_num_samples() == 1);
  
  SetConst(0).execute(m_weights_update.get());
  
  for(int sample=0; sample < top_error->get_num_samples(); sample++) {
    std::unique_ptr<Data> sample_top_error = top_error->get_samples_slice(sample, sample+1);
    
    sample_top_error->reshape(1, 1, m_num_filters, im2col.get_output_width(bottom_out));
    
    //gradient w.r.t. weights:
    //TODO: accumulate vs average?
    double batch_average = 1. / top_error->get_num_samples();
    MatrixMultiplicationMKL calculate_weight_gradient(sample_top_error.get(),
                                                      m_col_image.get(),
                                                      m_weights_update.get(),
                                                      MatrixMultiplication::MatrixOp::NoTranspose,
                                                      MatrixMultiplication::MatrixOp::MatrixTranspose,
                                                      1.//accumulate
                                                      );
    calculate_weight_gradient.execute();
    
    //gradient w.r.t. input:
    MatrixMultiplicationMKL calculate_input_gradient(m_weights.get(),
                                                     sample_top_error.get(),
                                                     m_col_image.get(),
                                                     MatrixMultiplication::MatrixOp::MatrixTranspose,
                                                     MatrixMultiplication::MatrixOp::NoTranspose);
    calculate_input_gradient.execute();
    
    std::unique_ptr<Data> sample_backprop_error = m_backprop_error->get_samples_slice(sample, sample+1);
    im2col.reverse(m_col_image.get(), sample_backprop_error.get());
  }

}

void ConvLayer::update(double learning_rate) {
  MatrixAdd apply_update(-1*learning_rate);
  apply_update.execute(m_weights.get(), m_weights_update.get());
  
//  m_weights_update->print();
  
  std::cout << "Sum of weight updates conv layer" << DataAbsSum().execute(m_weights_update.get()) << std::endl;
}

