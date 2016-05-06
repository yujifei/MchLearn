#pragma once
#include "../Vector/vector.h"
#include<vector>
#include<utility>

std::pair<Vector<float>, float > Perceptron(const std::vector<Vector<float> >& x, const std::vector<float>& y, float rate = 1.0f);
std::pair<Vector<float>, float > PerceptronDual(const std::vector<Vector<float> >& x, const std::vector<float>& y, float rate = 1.0f);
