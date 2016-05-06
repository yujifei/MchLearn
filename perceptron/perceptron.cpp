#include "perceptron.h"
#include "../Vector/Matrix.h"

#define EPSILON 0.0001f

float eval(const Vector<float>& w, const Vector<float>& x, float b)
{
    return w.dotProduct(x) + b;
}

bool check(const std::vector<Vector<float> >& x, const std::vector<float>& y)
{
    if (x.size() == 0 || y.size() == 0)
    {
        return false;
    }
    if (x.size() != y.size())
    {
        return false;
    }
    if (x.front().dim() == 0)
    {
        return false;
    }
    for (int i = 0; i < x.size(); ++i)
    {
        if (x.at(i).dim() != x.front().dim())
        {
            return false;
        }
    }

    return true;
}

std::pair<Vector<float>, float > Perceptron(const std::vector<Vector<float> >& x, const std::vector<float>& y, float rate /* = 1.0f */)
{
    if (!check(x, y))
    {
        return std::make_pair(Vector<float>(1, 0.0f), 0.0f);
    }

    Vector<float> w(x.front().dim(), 0.0f);
    float b = 0.0f;

    for (int i = 0; i < x.size();)
    {
        if (y[i] * eval(w, x.at(i), b) < EPSILON)
        {
            w += x.at(i) * (y[i] * rate);
            b += y[i] * rate;
            i = 0;
            continue;
        }
        ++i;
    }

    return std::make_pair(w, b);
}

std::pair<Vector<float>, float > PerceptronDual(const std::vector<Vector<float> >& x, const std::vector<float>& y, float rate)
{
    if (!check(x, y))
    {
        return std::make_pair(Vector<float>(1, 0.0f), 0.0f);
    }

    Matrix<float> gram(x.size(), x.size());
    for (int i = 0; i < x.size(); ++i)
    {
        for (int j = i; j < x.size(); ++j)
        {
            float r = x.at(i).dotProduct(x.at(j));
            gram(i, j) = r;
            gram(j, i) = r;
        }
    }

    Vector<float> alpha(x.size(), 0.0f);
    float b = 0.0f;

    for (int i = 0; i < x.size();)
    {
        float v = 0;
        for (int j = 0; j < x.size(); ++j)
        {
            v += alpha[j] * y[j] * gram(i, j);
        }
        if (y[i] * (v + b) < EPSILON)
        {
            alpha[i] += rate;
            b += rate * y[i];
            i = 0;
            continue;
        }
        ++i;
    }

    Vector<float> w(x.front().dim(), 0.0f);
    for (int i = 0; i < x.size(); ++i)
    {
        w += alpha[i] * y[i] * x[i];
    }

    return std::make_pair(w, b);
}
