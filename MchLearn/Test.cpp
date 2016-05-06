#include "../Vector/vector.h"
#include "../perceptron/perceptron.h"
#include<iostream>
#include<cstdlib>
#include<vector>

#define count_of(v) (sizeof(v) / sizeof(v[0]))

template<class T> void printv(char* msg, const Vector<T>& v)
{
    std::cout << msg << ':';
    if (v.dim() == 0)
    {
        std::cout << " empty\n";
        return;
    }
    std::cout << std::endl;
    for (int i = 0; i < v.dim() - 1; ++i)
    {
        std::cout << v[i] << ", ";
    }
    std::cout << v[v.dim() - 1] << std::endl;
}

void testv()
{
    float av1[] = {1.0, 2.7, 5.9, 3.4, 7.8};
    float av2[] = {3.2, 2.9, 8.1, 2.7, 2.5};
    Vector<float> v1(av1, count_of(av1));
    Vector<float> v2(av2, count_of(av2));
    Vector<float> v3(5, 1.4);

    printv("v1", v1);
    printv("v2", v2);
    printv("v1 * 4.5", v1 * 4.5);
    printv("v2 / 2.1", v2 / 2.1);
    printv("v1 + 5.6", v1 + 5.6);
    printv("v2 - 3.8", v2 - 3.8);
    printv("v1 ^ 3.0", v1 ^ 3.0);

    v1 *= 3.7;
    printv("after v1 *= 3.7, v1", v1);
    v2 /= 6.7;
    printv("after v2 /= 6.7, v2", v2);
    v1 += 7.4;
    printv("after v1 += 6.7, v1", v1);
    v2 -= 1.8;
    printv("after v2 -= 1.8, v2", v2);
    v1 ^= 0.5;
    printv("after v1 ^ 0.5, v1", v1);

    printv("v1 + v2", v1 + v2);
    printv("v1 - v2", v1 - v2);
    printv("v3", v3);
    v1 += v3;
    v2 -= v3;
    printv("after v1 -= v3", v1);
    printv("after v2 += v3", v2);

    printv("1.0 + v1", 1.0f + v1);
    printv("10.0 - v2", 10.0f - v2);
    printv("0.5 * v1", 0.5f * v1);

    std::cout << "accumulate of v1 " << v1.accumulate() << std::endl;
    std::cout << "dot product of v1 and v2 " << v1.dotProduct(v2) << std::endl;
}

void testperceptron()
{
    float av1[] = {3.0, 3.0};
    float av2[] = {4.0, 3.0};
    float av3[] = {1.0, 1.0};
    std::vector<Vector<float> > x;
    std::vector<float> y;
    x.push_back(Vector<float>(av1, count_of(av1)));
    x.push_back(Vector<float>(av2, count_of(av2)));
    x.push_back(Vector<float>(av3, count_of(av3)));
    y.push_back(1.0);
    y.push_back(1.0);
    y.push_back(-1.0);

    auto r1 = Perceptron(x, y);
    auto r2 = PerceptronDual(x, y);
    printv("w", r1.first);
    std::cout << "b: " << r1.second << std::endl;
    printv("w", r2.first);
    std::cout << "b: " << r2.second << std::endl;
}

int main(int argc, char* argv)
{
    testperceptron();
    system("pause");
    return 0;
}
