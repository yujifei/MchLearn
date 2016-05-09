#pragma once

#include "../Vector/vector.h"
#include <vector>

struct KdNode
{
    int dim;
    float lbound;
    float hbound;
    KdNode* left;
    KdNode* right;
    KdNode* parent;
    size_t  data;

    KdNode(): dim(0), lbound(0.0f), hbound(0.0f), left(nullptr), right(nullptr), parent(nullptr), data(0)
    {
    }
};

class KdTree
{
public:
    KdTree(): m_isBuilt(false), m_root(nullptr)
    {
    }
    ~KdTree()
    {
        destroy(m_root);
    }
    void build(const std::vector<Vector<float> >& data)
    {
        std::vector<size_t> indices;
        indices.reserve(data.size());
        for (size_t i = 0; i < data.size(); ++i)
        {
            indices.push_back(i);
        }
        m_root = buildTree(data, indices, NULL, 0, data.size(), 0);
    }
    void search(const Vector<float>& v, unsigned long n, std::vector<size_t>& idx);
    void reset()
    {
        destroy(m_root);
        m_root = nullptr;
    }

private:
    void destroy(KdNode* root);
    KdNode* buildTree(const std::vector<Vector<float> >& data, std::vector<size_t>& idx, KdNode* parent, size_t lb, size_t hb, size_t d);
    void partial(const std::vector<Vector<float> >& data, std::vector<size_t>& idx, size_t k, size_t lb, size_t hb, size_t d);
private:
    bool m_isBuilt;
    KdNode* m_root;
};