#include "perceptron.h"
#include "../Vector/Matrix.h"
#include <algorithm>
#include <fstream>

#define EPSILON 0.0001f

float eval(const Vector<float>& w, const Vector<float>& x, float b)
{
    return w.dotProduct(x) + b;
}

bool check(const std::vector<Vector<float> >& x, const std::vector<unsigned long>& y)
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

static void GetBatch(const std::vector<Vector<float> >& x, const std::vector<unsigned long>& y, const std::vector<unsigned long>& labels, std::vector<std::vector<Vector<float> > >& batches)
{
    for (auto i = 0; i < labels.size(); ++i)
    {
        auto& curBatch = batches[i];
        for (auto j = 0; j < x.size(); ++j)
        {
            if (y[j] == labels[i])
            {
                curBatch.emplace_back(std::move(x[j]));
            }
        }
    }
}

std::pair<Vector<float>, float > Perceptron(const std::vector<Vector<float> >& x, const std::vector<int>& y, unsigned long maxIter, unsigned long maxIterPerSample, float rate /* = 1.0f */)
{
    Vector<float> w(x.front().dim(), 0.0f);
    std::vector<int> iterPerSample(x.size(), 0);
    float b = 0.0f;

    for (size_t i = 0, k = 0; k < maxIter && i < x.size(); ++k)
    {
        if (y[i] * eval(w, x.at(i), b) < EPSILON && iterPerSample[i] < maxIterPerSample)
        {
            w += x.at(i) * (y[i] * rate);
            b += y[i] * rate;
            ++iterPerSample[i];
            i = 0;
            continue;
        }
        ++i;
    }

    return std::make_pair(w, b);
}

std::pair<Vector<float>, float > PerceptronDual(const std::vector<Vector<float> >& x, const std::vector<int>& y, unsigned long maxIter, unsigned long maxIterPerSample, float rate)
{
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
    std::vector<int> iterPerSample(x.size(), 0);

    for (size_t i = 0, k = 0; i < x.size() && k < maxIter; ++k)
    {
        float v = 0;
        for (int j = 0; j < x.size(); ++j)
        {
            v += alpha[j] * y[j] * gram(i, j);
        }
        if (y[i] * (v + b) < EPSILON && iterPerSample[i] < maxIterPerSample)
        {
            alpha[i] += rate;
            b += rate * y[i];
            ++iterPerSample[i];
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

//x1的标签为-1, x2标签为1
static std::pair<Vector<float>, float > Perceptron(const std::vector<Vector<float> >& x1, const std::vector<Vector<float>>& x2, unsigned long maxIter, unsigned long maxIterPerSample, float rate /* = 1.0f */)
{
    Vector<float> w(x1.front().dim(), 0.0f);
    std::vector<int> iterPerSample(x1.size() + x2.size(), 0);
    float b = 0.0f;

    for (size_t i = 0, k = 0; k < maxIter && i < x1.size() + x2.size(); ++k)
    {
        if (i < x1.size())
        {
            if (-eval(w, x1.at(i), b) < EPSILON && iterPerSample[i] < maxIterPerSample)
            {
                w -= x1.at(i) * rate;
                b -= rate;
                ++iterPerSample[i];
                i = 0;
                continue;
            }
            ++i;
        }
        else
        {
            if (eval(w, x2.at(i - x1.size()), b) < EPSILON && iterPerSample[i] < maxIterPerSample)
            {
                w += x2.at(i - x1.size()) * rate;
                b += rate;
                ++iterPerSample[i];
                i = 0;
                continue;
            }
            ++i;
        }
    }

    return std::make_pair(w, b);
}

static std::pair<Vector<float>, float > PerceptronDual(const std::vector<Vector<float> >& x1, const std::vector<Vector<float>>& x2, unsigned long maxIter, unsigned long maxIterPerSample, float rate)
{
    Matrix<float> gram(x1.size() + x2.size(), x1.size() + x2.size());
    for (int i = 0; i < x1.size() + x2.size(); ++i)
    {
        for (int j = i; j < x1.size() + x2.size(); ++j)
        {
            float r = 0.0f;
            if (i < x1.size())
            {
                if (j < x1.size())
                {
                    x1.at(i).dotProduct(x1.at(j));
                }
                else
                {
                    x1.at(i).dotProduct(x2.at(j - x1.size()));
                }
            }
            else
            {
                if (j < x1.size())
                {
                    x2.at(i - x1.size()).dotProduct(x1.at(j));
                }
                else
                {
                    x2.at(i - x1.size()).dotProduct(x2.at(j - x1.size()));
                }
            }
            gram(i, j) = r;
            gram(j, i) = r;
        }
    }

    Vector<float> alpha(x1.size() + x2.size(), 0.0f);
    float b = 0.0f;
    std::vector<int> iterPerSample(x1.size() + x2.size(), 0);

    for (size_t i = 0, k = 0; i < x1.size() && k < maxIter; ++ k)
    {
        float v = 0;
        float y = i < x1.size() ? -1 : 1;
        for (int j = 0; j < x1.size() + x2.size(); ++j)
        {
            v += alpha[j] * (j < x1.size() ? -1 : 1) * gram(i, j);
        }
        if (y * (v + b) < EPSILON && iterPerSample[i] < maxIterPerSample)
        {
            alpha[i] += rate;
            b += rate * y;
            i = 0;
            continue;
        }
        ++i;
    }

    Vector<float> w(x1.front().dim(), 0.0f);
    for (int i = 0; i < x1.size() + x2.size(); ++i)
    {
        w += alpha[i] * (i < x1.size() ? -1 : 1) * (i < x1.size() ? x1[i] : x2[i - x1.size()]);
    }

    return std::make_pair(w, b);
}

PerceptronModel* PerceptronTrain(const std::vector<Vector<float> >& x, const std::vector<unsigned long>& y, const PerceptronParam& param)
{
    if (!check(x, y))
    {
        return nullptr;
    }

    auto extLabel = std::minmax_element(y.begin(), y.end());
    std::vector<unsigned long> bucket(*extLabel.second - *extLabel.first + 1, 0);

    for (auto i = 0; i < y.size(); ++i)
    {
        ++bucket[y[i] - *extLabel.first];
    }

    PerceptronModel* model = new PerceptronModel;
    model->classLabels.reserve(bucket.size());
    model->classifiers.reserve(y.size() * (y.size() - 1) / 2);
    for (auto i = 0; i < bucket.size(); ++i)
    {
        if (bucket[i])
        {
            model->classLabels.push_back(i + *extLabel.first);
        }
    }

    std::vector<std::vector<Vector<float> > > batches(model->classLabels.size());
    for (auto i = 0; i < batches.size(); ++i)
    {
        batches[i].reserve(bucket[model->classLabels[i] - *extLabel.first]);
    }
    GetBatch(x, y, model->classLabels, batches);

    for (auto i = 0; i < model->classLabels.size() - 1; ++i)
    {
        for (auto j = i + 1; j < model->classLabels.size(); ++j)
        {
            auto result = Perceptron(batches[i], batches[j], param.maxIter, param.maxIterPerSample, param.rate);

            PerceptronModel::Classifier classifier;
            classifier.classLabelIdx1 = i;
            classifier.classLabelIdx2 = j;
            classifier.w = result.first;
            classifier.b = result.second;
            model->classifiers.push_back(classifier);
        }
    }
}

int GetClass(const PerceptronModel::Classifier& classifier, const Vector<float>& x)
{
    float v = eval(classifier.w, x, classifier.b);
    return (v > 0 ? 1 : -1);
}

unsigned long PerceptronPredict(const PerceptronModel& model, const Vector<float>& x)
{
    std::vector<int> vote(model.classLabels.size(), 0);
    for (auto i = 0; i < model.classifiers.size(); ++i)
    {
        int c = GetClass(model.classifiers[i], x);
        if (c < 0)
        {
            ++vote[model.classifiers[i].classLabelIdx1];
        }
        else
        {
            ++vote[model.classifiers[i].classLabelIdx2];
        }
    }

    int curMax = 0, curMaxIdx = 0;
    for (auto i = 0; i < vote.size(); ++i)
    {
        if (vote[i] > curMax)
        {
            curMax = vote[i];
            curMaxIdx = i;
        }
    }

    return model.classLabels[curMaxIdx];
}

bool IsValidModel(const PerceptronModel& model)
{
    if (model.classLabels.empty())
    {
        return false;
    }
    
    for (auto i = 0; i < model.classLabels.size() - 1; ++i)
    {
        if (model.classLabels[i] >= model.classLabels[i + 1])
        {
            return false;
        }
    }

    std::vector<bool> bucket(model.classLabels.back() - model.classLabels.front() + 1, false);
    for (auto i = 0; i < model.classLabels.size(); ++i)
    {
        if (!bucket[model.classLabels[i]])
        {
            bucket[model.classLabels[i]] = true;
        }
        else
        {
            return false;
        }
    }

    if (model.classifiers.size() != model.classLabels.size())
    {
        return false;
    }
    if (model.classifiers.front().w.dim() == 0)
    {
        return false;
    }

    for (auto i = 0; i < model.classifiers.size(); ++i)
    {
        if (model.classifiers[i].classLabelIdx1 >= model.classLabels.size() || model.classifiers[i].classLabelIdx2 >= model.classLabels.size()
            || model.classifiers[i].classLabelIdx1 >= model.classifiers[i].classLabelIdx2)
        {
            return false;
        }
        if (model.classifiers[i].w.dim() != model.classifiers.front().w.dim())
        {
            return false;
        }
    }

    return true;
}

bool SavePerceptronModel(const PerceptronModel& model, const std::string& fileName)
{
    if (!IsValidModel(model))
    {
        return false;
    }

    std::ofstream ofs(fileName, std::ios::binary | std::ios::out);
    ofs << model.classLabels.size() << model.classifiers.front().w.dim();
    for (auto i = 0; i < model.classLabels.size(); ++i)
    {
        ofs << model.classLabels[i];
    }
    for (auto i = 0; i < model.classifiers.size(); ++i)
    {
        auto& cur = model.classifiers[i];
        ofs << cur.classLabelIdx1 << cur.classLabelIdx2;
        for (auto j = 0; j < cur.w.dim(); ++j)
        {
            ofs << cur.w[j];
        }
        ofs << cur.b;
    }

    ofs.flush();
    ofs.close();
    return true;
}

PerceptronModel* LoadPerceptronModel(const std::string& fileName)
{
    std::ifstream ifs(fileName, std::ios::binary | std::ios::in);

    PerceptronModel* model = new PerceptronModel;
    if (!model)
    {
        return nullptr;
    }

    size_t nLabel = 0, dim = 0;
    ifs >> nLabel >> dim;
    model->classLabels.reserve(nLabel);
    model->classifiers.reserve(nLabel);
    for (auto i = 0; i < nLabel; ++i)
    {
        unsigned long c = 0;
        ifs >> c;
        model->classLabels.push_back(c);
    }

    for (auto i = 0; i < nLabel; ++i)
    {
        model->classifiers.emplace_back(dim);
        auto& cur = model->classifiers.back();
        for (auto j = 0; j < dim; ++j)
        {
            ifs >> cur.w[j];
        }
        ifs >> cur.b;
    }

    if (!IsValidModel(*model))
    {
        delete model;
        return nullptr;
    }
    return model;
}