//
//  metrics.h
//  nntest
//
//  Created by Tobias Domhan on 1/7/14.
//  Copyright (c) 2014 Tobias Domhan. All rights reserved.
//

#ifndef __nntest__metrics__
#define __nntest__metrics__

#include <iostream>
#include "data.h"


/*
 * predictions: one-hot encoded predictions
 * labels: one-hot encoded labels
 */
double accuracy(Data* predictions, Data* labels);

double accuracy(const std::vector<Data*> &predictions, const std::vector<Data*> &labels);

#endif /* defined(__nntest__metrics__) */
