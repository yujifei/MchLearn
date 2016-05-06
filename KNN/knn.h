#pragma once
#include "../Vector/vector.h"
#include "../Utilities/types.h"
#include <vector>

class KNN
{
public:
    KNN(): m_hasKdTree(false)
    {
    }
    bool append(Vector<float>& x, float y);
    void reset();
    bool search(const Vector<float>& v, size_t n, std::vector<size_t>& idx);
    const Vector<float>& getX(size_t i)
    {
        return m_x.at(i);
    }
    float getY(size_t i)
    {
        return m_y.at(i);
    }
private:
    bool searchDirectly(const Vector<float>& v, size_t n, std::vector<size_t>& idx);
    bool searchKdTree(const Vector<float>& v, size_t n, std::vector<size_t>& idx);
    bool buildKdTree();
private:
    std::vector<Vector<float> > m_x;
    std::vector<float> m_y;
    bool m_hasKdTree;
};