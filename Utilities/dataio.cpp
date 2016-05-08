#include "dataio.h"
#include "../libzlib/zlib.h"

int reverseInt(int v)
{
    unsigned char b1, b2, b3, b4;
    b1 = v & 255;
    b2 = (v >> 8) & 255;
    b3 = (v >> 16) & 255;
    b4 = v >> 24;
    return ((b1 << 24) | (b2 << 16) | (b3 << 8) | b4);
}

bool loadMnistImage(const char* fileName, std::vector<Vector<float> >& data, size_t ndata)
{
    gzFile file = gzopen(fileName, "rb");

    int magicNum;
    gzread(file, &magicNum, sizeof(magicNum));
    magicNum = reverseInt(magicNum);
    if (magicNum != 2051)
    {
        gzclose(file);
        return false;
    }

    int imageNumber;
    gzread(file, &imageNumber, sizeof(imageNumber));
    imageNumber = reverseInt(imageNumber);
    imageNumber = ndata ? ndata : imageNumber;
    data.reserve(imageNumber);

    int imgRow, imgCol;
    gzread(file, &imgRow, sizeof(imgRow));
    gzread(file, &imgCol, sizeof(imgCol));
    imgRow = reverseInt(imgRow);
    imgCol = reverseInt(imgCol);
    int imgSize = imgRow * imgCol;

    for (int i = 0; i < imageNumber; ++i)
    {
        data.emplace_back(Vector<float>(imgSize));
        auto& cur = data.back();
        for (int j = 0; j < imgSize; ++j)
        {
            unsigned char pixel;
            gzread(file, &pixel, sizeof(pixel));
            cur[j] = (float)pixel;
        }
    }

    gzclose(file);
    return true;
}

bool loadMnistLabel(const char* fileName, std::vector<float>& data, size_t ndata)
{
    gzFile file = gzopen(fileName, "rb");

    int magicNum;
    gzread(file, &magicNum, sizeof(magicNum));
    magicNum = reverseInt(magicNum);
    if (magicNum != 2049)
    {
        gzclose(file);
        return false;
    }

    int nLabel;
    gzread(file, &nLabel, sizeof(nLabel));
    nLabel = reverseInt(nLabel);
    nLabel = ndata ? ndata : nLabel;
    data.reserve(nLabel);

    for (int i = 0; i < nLabel; ++i)
    {
        unsigned char l;
        gzread(file, &l, sizeof(l));
        data.push_back((float)l);
    }
    
    gzclose(file);
    return true;
}