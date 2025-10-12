#include <iostream>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <string>
#include <vector>
#include "image_ops.h"
#include <chrono>
using namespace std;
//g++ main.cpp image_operation_functions.cpp -o out

void write_output_rgb(const vector<unsigned char>& red, const vector<unsigned char>& green, const vector<unsigned char>& blue, int width, int height, const string& output_file="assets/rgb_img.jpg"){
    size_t dimension = static_cast<size_t>(width) * static_cast<size_t>(height);
    vector<unsigned char> combined_channels(dimension * 3);
    for(size_t i = 0; i<dimension; ++i){
        combined_channels[i*3 + 0] = red[i];
        combined_channels[i*3 + 1] = green[i];
        combined_channels[i*3 + 2] = blue[i];
    }
    cout << "combined_channels[1]: " << static_cast<int>(combined_channels[1]) << endl;
    int success = stbi_write_jpg(output_file.c_str(), width, height, 3, combined_channels.data(), 90);
    cout << "Wrote RGB Image? " << success << endl;
}

void write_output_gray(const vector<unsigned char>& gray, int width, int height, const string& output_file = "assets/gradients.jpg"){
    int success = stbi_write_jpg(output_file.c_str(), width, height, 1, gray.data(), 90);
    cout << "Wrote Grayscale Image? " << success << endl;
}

int main(int argc, char** argv){
    // Accept optional input and output paths
    string filepath = (argc > 1) ? argv[1] : string("assets/image.jpg");
    string outpath = (argc > 2) ? argv[2] : string("assets/gradients.jpg");

    int height = 0, width = 0, channels = 0;
    //last parameter is number of desired channels. 3 is RGB
    cout << "Loading File: " << filepath << endl;
    unsigned char *data = stbi_load(filepath.c_str(), &width, &height, &channels, 3);
    if(data){
        cout << "width: " << width << endl;
        cout << "height: " << height << endl;
        cout << "channels: " << channels << endl;
        
        size_t dim = static_cast<size_t>(width) * static_cast<size_t>(height);
        vector<unsigned char> r(dim);
        vector<unsigned char> g(dim);
        vector<unsigned char> b(dim);

        for(size_t i = 0; i < dim; ++i){
            r[i] = data[i*3 + 0];
            g[i] = data[i*3 + 1];
            b[i] = data[i*3 + 2];
        }
        stbi_image_free(data); //frees allocated image memory

        auto t0 = chrono::high_resolution_clock::now();
        vector<unsigned char> output = sobel_convolve(r,g,b, width, height, 1.4);
        auto t1 = chrono::high_resolution_clock::now();
        auto ms = chrono::duration_cast<chrono::milliseconds>(t1 - t0).count();

        write_output_gray(output, width, height, outpath);
        cout << "PROCESS_TIME_MS: " << ms << endl;
    }else{
        cout << "Failure loading image" << endl;
        cout << "Reason: " << stbi_failure_reason() << endl;
        return 2;
    }
    return 0;
    
}

