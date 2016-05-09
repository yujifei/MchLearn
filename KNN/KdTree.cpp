#include "KdTree.h"

void KdTree::destroy(KdNode* root)
{
    if (!root)
    {
        return;
    }
    destroy(root->left);
    destroy(root->right);
    delete root;
}

KdNode* KdTree::buildTree(const std::vector<Vector<float> >& data, std::vector<size_t>& idx, KdNode* parent, size_t lb, size_t hb, size_t d)
{
    if (lb == hb)
    {
        return nullptr;
    }

    KdNode* cur = new KdNode;
    cur->dim = d;
    cur->parent = parent;
    cur->lbound = data[idx[lb]][d];
    cur->hbound = data[idx[lb]][d];
    cur->data = idx[lb];

    if (lb == hb - 1)
    {
        cur->left = nullptr;
        cur->right = nullptr;
        return cur;
    }

    for (size_t i = lb; i < hb; ++i)
    {
        if (data[idx[i]][d] < cur->lbound)
        {
            cur->lbound = data[idx[i]][d];
        }
        if (data[idx[i]][d] > cur->hbound)
        {
            cur->hbound = data[idx[i]][d];
        }
    }

    size_t pivot = (lb + hb) / 2;
    partial(data, idx, pivot, lb, hb, d);
    cur->data = idx[pivot];

    cur->left = buildTree(data, idx, cur, lb, pivot, (d + 1) % data.front().dim());
    cur->right = buildTree(data, idx, cur, pivot + 1, hb, (d + 1) % data.front().dim());
}

void KdTree::partial(const std::vector<Vector<float> >& data, std::vector<size_t>& idx, size_t k, size_t lb, size_t hb, size_t d)
{
    if (lb == hb - 1)
    {
        return;
    }

    int pivot = (lb + hb) / 2;
    std::swap(idx[pivot], idx[hb - 1]);

    int i = lb - 1;
    for (int j = lb; j < hb - 1; ++j)
    {
        if (data[idx[j]][d] < data[idx[hb - 1]][d])
        {
            ++i;
            std::swap(idx[i], idx[j]);
        }
    }
    ++i;
    std::swap(idx[i], idx[hb - 1]);

    if (i > k)
    {
        partial(data, idx, k, lb, i, d);
    }
    else if (i < k)
    {
        partial(data, idx, k - i - 1, lb + 1, hb, d);
    }
}

void KdTree::search(const Vector<float>& v, unsigned long n, std::vector<size_t>& idx)
{

}