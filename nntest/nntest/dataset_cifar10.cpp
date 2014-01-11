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

#include <sstream>

using namespace std;

DataSetCIFAR10::DataSetCIFAR10(const std::string &data_folder, int test_fold, int batch_size) :
 DataSet(batch_size) {
   m_has_test_data = true;

   const int NUM_FOLDS = 5;
   std::vector<std::unique_ptr<Data>> train_data;
   std::vector<std::unique_ptr<Data>> train_labels;
   std::vector<Data*> train_data_raw;
   std::vector<Data*> train_labels_raw;
   
   for(int fold=1; fold < NUM_FOLDS+1; fold++) {
     std::stringstream fname;
     fname << data_folder << "/cifar-10-batches-bin/data_batch_" << fold << ".bin";
     DataSetCIFAR10File data_file(fname.str());
     
     if(fold != test_fold) {
       train_data.push_back(data_file.get_data_permenantly());
       train_labels.push_back(data_file.get_labels_permenantly());
       train_data_raw.push_back(train_data.back().get());
       train_labels_raw.push_back(train_labels.back().get());
     } else {
       m_test_data = data_file.get_data_permenantly();
       m_test_labels = data_file.get_labels_permenantly();
     }
   }
   if(test_fold == 0) {
     std::stringstream fname;
     fname << data_folder << "/cifar-10-batches-bin/test_batch.bin";
     DataSetCIFAR10File data_file(fname.str());
     
     m_test_data = data_file.get_data_permenantly();
     m_test_labels = data_file.get_labels_permenantly();
   }
   
   DataCPU data(0,0);
   m_train_data = data.vstack(train_data_raw);
   m_train_labels = data.vstack(train_labels_raw);
   
   std::cout << "Train data shape: " << m_train_data->get_size_dim(0) << " " <<  m_train_data->get_size_dim(1) << std::endl;
   
   train_data.clear();
   train_labels.clear();
  
   setup();
}


/**
 Format of the datafile
 <1 x label><3072 x pixel>
 ...
 <1 x label><3072 x pixel>
 */
void DataSetCIFAR10File::read_file(const std::string &fname) {
  ifstream infile (fname, ios::in | ios::binary);
  const int buffer_size = 3073;
  const int num_pixels = 3072;
  const int num_classes = 10;
  unsigned char buffer[buffer_size];
  
  if(!infile.is_open()) {
    cout << "Unable to open file " << fname << endl;
    throw new runtime_error("Unable to open file");
  }
  long num_samples = 0;
  while (infile.read((char*)&buffer[0], buffer_size)) {
    num_samples++;
  }
  cout << "Number of samples: " << num_samples << endl;
  
  assert(num_samples > 0);
  
  //reset file pointer
  infile.clear();
  infile.seekg(0, ios::beg);
  
  m_data = std::unique_ptr<Data>(new DataCPU(num_samples, num_pixels));
  m_labels = std::unique_ptr<Data>(new DataCPU(num_samples, num_classes));
  SetConst(0).execute(m_labels.get());
  
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
}
