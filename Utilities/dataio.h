#pragma once
#include "types.h"
#include "../Vector/vector.h"
#include <vector>

bool loadMnistImage(const char* fileName, std::vector<Vector<float> >& data, size_t ndata = 0);
bool loadMnistLabel(const char* fileName, std::vector<float>& data, size_t ndata = 0);
