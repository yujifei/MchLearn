#pragma once
#include <memory>
template<class T> class Matrix
{
public:
    Matrix(size_t r, size_t c, T v = T());
    Matrix(size_t r, size_t c, const T* av, size_t l, T v = T());
    Matrix(const Matrix& m);
    Matrix& operator=(const Matrix& m);
    T& operator()(size_t r, size_t c)
    {
        return m_data[r * m_col + c];
    }
    const T& operator()(size_t r, size_t c) const
    {
        return m_data[r * m_col + c];
    }
    size_t row() const
    {
        return m_row;
    }
    size_t col() const
    {
        return m_col;
    }
private:
    size_t m_row;
    size_t m_col;
    std::unique_ptr<T[]> m_data;
};

template<class T> Matrix<T>::Matrix(size_t r, size_t c, T v /* = T() */)
{
    m_data.reset(new T[r * c]);
    m_row = r;
    m_col = c;
    for (int y = 0; y < r; ++y)
    {
        for (int x = 0; x < c; ++x)
        {
            m_data[y * c + x] = v;
        }
    }
}
template<class T> Matrix<T>::Matrix(size_t r, size_t c, const T* av, size_t l, T v /* = T() */)
{
    m_data.reset(new T[r * c]);
    m_row = r;
    m_col = c;
    int s = std::min(r * c, l);
    for (int i = 0; i < s; ++i)
    {
        m_data[i] = av[i];
    }
    for (int i = s; i < r * c; ++i)
    {
        m_data[i] = v;
    }
}

template<class T> Matrix<T>::Matrix(const Matrix& m)
{
    m_data.reset(new T[m.m_col * m.m_row]);
    m_row = m.m_row;
    m_col = m.m_col;
    memcpy(m_data.get(), m.m_data.get, sizeof(T) * m.m_col * m_row);
}

template<class T> Matrix<T>& Matrix<T>::operator=(const Matrix<T>& m)
{
    if (this == &m)
    {
        return *this;
    }
    m_data.reset(new T[m.m_col * m.m_row]);
    m_row = m.m_row;
    m_col = m.m_col;
    memcpy(m_data.get(), m.m_data.get, sizeof(T) * m.m_col * m_row);
    return *this;
}