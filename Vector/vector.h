#pragma once

#include "../Utilities/types.h"
#include<memory>
#include<iostream>

template<class T> class Vector
{
public:
    Vector(): m_size(0)
    {
        m_data.reset(nullptr);
    }
    Vector(size_t n, T v = T());
    Vector(size_t n, const T* v, size_t nv, T dv = T());
    Vector(const T* v, size_t n);
    Vector(const Vector& v);
    Vector(Vector&& v);
    ~Vector(){}
    Vector& operator=(const Vector& v);
    Vector& operator=(Vector&& v);
    const T& operator[](size_t i) const
    {
        return m_data[i];
    }
    T& operator[](size_t i)
    {
        return m_data[i];
    }
    bool resize(size_t n)
    {
        if (n > m_data)
        {
            T* data = new T[n];
            if (!data)
            {
                return false;
            }
            for (size_t i = 0; i < m_size; ++i)
            {
                data[i] = m_data[i];
            }
            m_data.reset(data);
        }
        return true;
    }

    Vector operator+(T v) const;
    Vector operator-(T v) const;
    Vector operator*(T v) const;
    Vector operator/(T v) const;
    Vector operator^(T v) const;


    void operator+=(T v);
    void operator-=(T v);
    void operator*=(T v);
    void operator/=(T v);
    void operator^=(T v);


    Vector operator+(const Vector& v) const;
    Vector operator-(const Vector& v) const;
    void operator+=(const Vector& v);
    void operator-=(const Vector& v);

    T accumulate() const;
    T dotProduct(const Vector& v) const;
    T dist(const Vector& v) const;
    template<class OP> void apply(const OP& op)
    {
        for (int i = 0; i < m_size; ++i)
        {
            m_data[i] = op(m_data[i]);
        }
    }

    size_t dim() const
    {
        return m_size;
    }

private:
    size_t m_size;
    std::unique_ptr<T[]> m_data;
};

template<class T> Vector<T>::Vector(size_t n, T v)
{
    m_data.reset(new T[n]);
    for (size_t i = 0; i < n; ++i)
    {
        m_data[i] = v;
    }
    m_size = n;
}

template<class T> Vector<T>::Vector(size_t n, const T* v, size_t nv, T dv /* = T() */)
{
    m_data.reset(new T[n]);
    size_t l = std::min(n, nv);
    for (int i = 0; i < l; ++i)
    {
        m_data[i] = v[i];
    }

    for (int i = l; i < n; ++i)
    {
        m_data[i] = dv;
    }
    m_size = n;
}

template<class T> Vector<T>::Vector(const T* v, size_t n)
{
    m_data.reset(new T[n]);
    for (int i = 0; i < n; ++i)
    {
        m_data[i] = v[i];
    }
    m_size = n;
}

template<class T> Vector<T>::Vector(const Vector<T>& v)
{
    if (v.m_size == 0)
    {
        return;
    }
    m_data.reset(new T[v.m_size]);
    for (int i = 0; i < v.m_size; ++i)
    {
        m_data[i] = v[i];
    }
    m_size = v.m_size;
}

template<class T> Vector<T>::Vector(Vector<T>&& v)
{
    m_data = std::move(v.m_data);
    m_size = v.m_size;
}

template<class T> Vector<T>& Vector<T>::operator=(const Vector<T>& v)
{
    if (this == &v)
    {
        return *this;
    }
    
    m_data.reset(NULL);
    m_size = v.dim();

    if (v.dim > 0)
    {
        m_data.reset(new T[v.dim()]);
        for (int i = 0; i < m_size; ++i)
        {
            m_data[i] = v[i];
        }
    }
    return *this;
}

template<class T> Vector<T>& Vector<T>::operator=(Vector<T>&& v)
{
    if (this == &v)
    {
        return *this;
    }
    m_data = std::move(v.m_data);
    m_size = v.m_size;
}

template<class T> Vector<T> Vector<T>::operator+(T v) const
{
    Vector<T> r = *this;
    r += v;
    return r;
}

template<class T> Vector<T> Vector<T>::operator-(T v) const
{
    Vector<T> r = *this;
    r -= v;
    return r;
}

template<class T> Vector<T> Vector<T>::operator*(T v) const
{
    Vector<T> r = *this;
    r *= v;
    return r;
}

template<class T> Vector<T> Vector<T>::operator/(T v) const
{
    Vector<T> r = *this;
    r /= v;
    return r;
}

template<class T> Vector<T> Vector<T>::operator^(T v) const
{
    Vector<T> r = *this;
    r ^= v;
    return r;
}

template<class T> void Vector<T>::operator+=(T v)
{
    for (int i = 0; i < m_size; ++i)
    {
        m_data[i] += v;
    }
}

template<class T> void Vector<T>::operator-=(T v)
{
    for (int i = 0; i < m_size; ++i)
    {
        m_data[i] -= v;
    }
}

template<class T> void Vector<T>::operator*=(T v)
{
    for (int i = 0; i < m_size; ++i)
    {
        m_data[i] *= v;
    }
}

template<class T> void Vector<T>::operator/=(T v)
{
    for (int i = 0; i < m_size; ++i)
    {
        m_data[i] /= v;
    }
}

template<class T> void Vector<T>::operator^=(T v)
{
    for (int i = 0; i < m_size; ++i)
    {
        std::pow(m_data[i], v);
    }
}

template<class T> Vector<T> Vector<T>::operator+(const Vector<T>& v) const
{
    Vector<T> r = *this;
    r += v;
    return r;
}

template<class T> Vector<T> Vector<T>::operator-(const Vector<T>& v) const
{
    Vector<T> r = *this;
    r -= v;
    return r;
}

template<class T> void Vector<T>::operator+=(const Vector<T>& v)
{
    if (m_size != v.m_size)
    {
        return;
    }
    for (int i = 0; i < m_size; ++i)
    {
        m_data[i] += v[i];
    }
}

template<class T> void Vector<T>::operator-=(const Vector<T>& v)
{
    if (m_size != v.m_size)
    {
        return;
    }
    for (int i = 0; i < m_size; ++i)
    {
        m_data[i] -= v[i];
    }
}

template<class T> T Vector<T>::accumulate() const
{
    T rst = T();
    for (int i = 0; i < m_size; ++i)
    {
        rst += m_data[i];
    }
    return rst;
}

template<class T> T Vector<T>::dotProduct(const Vector<T>& v) const
{
    T rst = T();
    for (int i = 0; i < m_size; ++i)
    {
        rst += m_data[i] * v[i];
    }
    return rst;
}
template<class T> T Vector<T>::dist(const Vector<T>& v) const
{
    T rst = T();
    for (int i = 0; i < m_size; ++i)
    {
        T tmp = m_data[i] - v[i]
        rst += tmp * tmp;
    }
    return rst;
}
template<class T> Vector<T> operator+(T v, const Vector<T>& vc)
{
    return vc + v;
}
template<class T> Vector<T> operator-(T v, const Vector<T>& vc)
{
    auto vt = vc;
    for(int i = 0; i < vt.dim(); ++i)
    {
        vt[i] = v - vt[i];
    }
    return vt;
}
template<class T> Vector<T> operator*(T v, const Vector<T>& vc)
{
    return vc * v;
}