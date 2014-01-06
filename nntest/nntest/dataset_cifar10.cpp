//
//  dataset_cifar10.cpp
//  nntest
//
//  Created by Tobias Domhan on 1/5/14.
//  Copyright (c) 2014 Tobias Domhan. All rights reserved.
//

#include "dataset_cifar10.h"

#include <iostream>
#include <fstream>
#include <exception>
#include <vector>

#include "data_cpu.h"
#include "math_util.h"

using namespace std;

DataSetCIFAR10::DataSetCIFAR10(const char* fname, int batch_size) :
 DataSet(batch_size) {
  read_file(fname);
  setup();
}


/**
 Format of the datafile
 <1 x label><3072 x pixel>
 ...
 <1 x label><3072 x pixel>
 */
void DataSetCIFAR10::read_file(const char* fname) {
  ifstream infile (fname, ios::in | ios::binary);
  const int buffer_size = 3073;
  const int num_pixels = 3072;
  const int num_classes = 10;
  unsigned char buffer[buffer_size];
  
  if(!infile.is_open()) {
    cout << "Unable to open file " << fname << endl;
    throw new runtime_error("Unable to open file");
    return;
  }
  long num_samples = 0;
  while (infile.read((char*)&buffer[0], buffer_size)) {
    num_samples++;
  }
  cout << "Number of samples: " << num_samples << endl;
  
  //reset file pointer
  infile.clear();
  infile.seekg(0, ios::beg);
  
  m_data = new DataCPU(num_samples, num_pixels);
  m_labels = new DataCPU(num_samples, num_classes);
  SetConst(0).execute(m_labels);
  
  long row_index = 0;
  while (infile.read((char*)&buffer[0], buffer_size)) {
    int label = (int)buffer[0];
    //one-hot encoding
    m_labels->get_data()[m_labels->get_index(row_index, label)] = 1;
    
    //read pixel values
    for (int i=0; i<num_pixels; i++) {
      unsigned char pixel = buffer[1+i];
      m_data->get_data()[m_data->get_index(row_index, i)] = pixel/255.;
    }
    
    row_index++;
  }
  //m_labels->get_rows_slice(0,11)->print();
}
