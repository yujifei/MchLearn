#pragma once
#include "../Vector/vector.h"
#include<vector>
#include<string>
#include<utility>

struct PerceptronModel
{
    struct Classifier
    {
        unsigned long classLabelIdx1;
        unsigned long classLabelIdx2;
        Vector<float> w;
        float b;
    };
    std::vector<unsigned long> classLabels;
    std::vector<Classifier> classifiers;
};

struct PerceptronParam
{
    unsigned long maxIter;
    unsigned long maxIterPerSample;
    float rate;
};

std::pair<Vector<float>, float > Perceptron(const std::vector<Vector<float> >& x, const std::vector<int>& y, unsigned long maxIter, unsigned long maxIterPerSample = 10, float rate = 1.0f);
std::pair<Vector<float>, float > PerceptronDual(const std::vector<Vector<float> >& x, const std::vector<int>& y, unsigned long maxIter, unsigned long maxIterPerSample = 10, float rate = 1.0f);

PerceptronModel* PerceptronTrain(const std::vector<Vector<float> >& x, const std::vector<unsigned long>& y, const PerceptronParam& param);
unsigned long PerceptronPredict(const PerceptronModel& model, const Vector<float>& x);
bool SavePerceptronModel(const PerceptronModel& modle, const std::string& fileName);
PerceptronModel* LoadPerceptronModel(const std::string& fileName);