#ifndef IMAGE_OPERATIONS
#define IMAGE_OPERATIONS
#include <vector>
std::vector<unsigned char> grayscale(const std::vector<unsigned char>& red, const std::vector<unsigned char>& green, const std::vector<unsigned char>& blue, int width, int height);
std::vector<unsigned char> gaussian_convolve(const std::vector<unsigned char>& red, const std::vector<unsigned char>& green, const std::vector<unsigned char>& blue,int width, int height, double std_dev);
#endif