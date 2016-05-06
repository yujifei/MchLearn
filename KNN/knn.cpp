#include "knn.h"
#include<cstdlib>
#include<ctime>

void findFirstK(std::vector<float>& value, std::vector<size_t>& index, size_t start, size_t end, size_t k)
{
    if (start == end)
    {
        return;
    }

    srand(time(0));
    KNN::size_t pivot = rand() % (end - start + 1);
    std::swap(value[pivot], value[end]);
    std::swap(index[pivot], index[end]);

    int i = -1;
    for (int j = 0; j < end; ++j)
    {
        if (value[j] < value[end])
        {
            ++i;
            if (i != j)
            {
                std::swap(value[i], value[j]);
                std::swap(index[i], index[j]);
            }
        }
    }
    ++i;
    if (i != end)
    {
        std::swap(value[i], value[end]);
        std::swap(index[i], index[end]);
    }

    if (i == k)
    {
        return;
    }
    if (i > k)
    {
        findFirstK(value, index, start, i - 1, k);
    }
    else
    {
        findFirstK(value, index, i + 1, end, i - k - 1);
    }
}

bool KNN::append(Vector<float>& x, float y)
{
    if (m_x.empty() && x.dim() == 0)
    {
        return false;
    }
    if (!m_x.empty() && x.dim() != m_x.front().dim())
    {
        return false;
    }
    m_x.push_back(x);
    m_y.push_back(y);
    return true;
}

void KNN::reset()
{
    m_x.clear();
    m_y.clear();
}

bool KNN::search(const Vector<float>& v, size_t n, std::vector<size_t>& idx)
{
    if (n == 0 || n > m_x.size())
    {
        return false;
    }
    if (v.dim() != m_x.front().dim())
    {
        return false;
    }
    if (m_x.size() >= 3 * m_x.front().dim())
    {
        if (!m_hasKdTree)
        {
            m_hasKdTree = buildKdTree();
        }
        if (m_hasKdTree)
        {
            return (searchKdTree(v, n, idx));
        }
    }
    return searchDirectly(v, n, idx);
}

bool KNN::buildKdTree()
{
    return false;
}

bool KNN::searchKdTree(const Vector<float>& v, size_t n, std::vector<size_t>& idx)
{
    return false;
}

bool KNN::searchDirectly(const Vector<float>& v, size_t n, std::vector<size_t>& idx)
{
    std::vector<float> dist(m_x.size());
    std::vector<size_t> disti(m_x.size());
    for (size_t i = 0; i < m_x.size(); ++i)
    {
        dist[i] = m_x[i].dist(v);
        disti[i] = i;
    }

    findFirstK(dist, disti, 0, m_x.size() - 1, n - 1);
    idx.assign(disti.begin(), disti.begin() + n);
}