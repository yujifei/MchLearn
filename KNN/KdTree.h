#pragma once

#include "../Vector/vector.h"
#include <vector>

struct KdNode
{
    size_t dataIdx;
    int dim;
    float lbound;
    float hbound;
    KdNode* left;
    KdNode* right;
    KdNode* parent;

    KdNode(): dataIdx(0), dim(0), lbound(0.0f), hbound(0.0f), left(nullptr), right(nullptr), parent(nullptr)
    {
    }
};

class KdTree
{
public:
    KdTree(): m_isBuilt(false), m_head(nullptr)
    {
    }
    ~KdTree()
    {
        destroy();
    }
    void build(const std::vector<Vector<float> >& data);
    void search(const Vector<float>& v, unsigned long n, std::vector<size_t>& idx);
    void reset()
    {
        destroy();
    }

private:
    void destroy();
    KdNode* buildTree(const std::vector<Vector<float> >& data, std::vector<size_t>& idx, size_t lb, size_t hb, size_t d);
    void patial(const std::vector<Vector<float> >& data, std::vector<size_t>& idx, size_t lb, size_t hb, size_t d);
private:
    bool m_isBuilt;
    KdNode* m_head;
};