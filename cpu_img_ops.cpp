#include "image_ops.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <complex>
#include <chrono>
#include <thread>
#include <functional>
#include <execution>
using namespace std;


//timing var speeds
long long int grayscale_speed;//
long long int fft_speed;//
long long int gaussian_kernel_gen_speed;//
long long int pad_speed = 0;//
long long transpose_speed = 0;//
long long back_convert_speed;//
long long int ifft_speed;//
long long int mult_speed;//

//returns a simple grayscale image by averaging the three channels,
//takes in the three channels, and width and height as input
//Multithread this process
void grayscale(vector<unsigned char>& out, const vector<unsigned char>& red, const vector<unsigned char>& green, const vector<unsigned char>& blue, int width, int height, int start, int end){
    size_t dimension = end-start;
    for(size_t i = start; i<end; ++i){
        int average =  (static_cast<int>(red[i]) + static_cast<int>(green[i]) + static_cast<int>(blue[i])) /3 ;
        out[i] = static_cast<unsigned char>(average);
    }
}
vector<unsigned char> grayscale_thread_mg(const vector<unsigned char>& red, const vector<unsigned char>& green, const vector<unsigned char>& blue, int width, int height){
    
    int num_threads = thread::hardware_concurrency();
    if(num_threads == 0)num_threads =16;
    cout << "threads: " << num_threads << endl;
    const int block_size = (width*height) / num_threads;
    vector<unsigned char> out(width*height);
    vector<thread> threads;
    for(int i = 0; i< num_threads; i++){
        int start = i*block_size;
        int end = (i == num_threads - 1) ? width*height : (i+1)*block_size;
        threads.emplace_back([&out, &red, &green, &blue, width, height, start, end]() {
            grayscale(out, red, green, blue, width, height, start, end);
        });
    }
    for (std::thread& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }
    
    
    cout << "Finished Grayscale Conversion" << endl;

    return out;
}
//1D FFT
vector<complex<double>> fft(const vector<complex<double>>& matrix) {
    size_t n = matrix.size();
    if (n <= 1) return matrix;

    vector<complex<double>> evens(n / 2);
    vector<complex<double>> odds(n / 2);
    for (size_t i = 0; i < n / 2; i++) {
        evens[i] = matrix[2 * i];
        odds[i] = matrix[2 * i + 1];
    }

    vector<complex<double>> e_out = fft(evens);
    vector<complex<double>> o_out = fft(odds);

    vector<complex<double>> y(n);
    //Using a running product is much faster than calling pow() in a loop.
    complex<double> omega(1.0, 0.0);
    complex<double> omega_n = polar(1.0, -2.0 * 3.14 / n); 

    for (size_t j = 0; j < n / 2; ++j) {
        complex<double> t = omega * o_out[j];
        y[j] = e_out[j] + t;
        y[j + n / 2] = e_out[j] - t;
        omega *= omega_n; // Update omega for the next iteration
    }
    return y;
}

//1D IFFT
vector<complex<double>> ifft(const vector<complex<double>>& matrix) {
    size_t n = matrix.size();
    if (n <= 1) return matrix;

    vector<complex<double>> evens(n / 2);
    vector<complex<double>> odds(n / 2);
    for (size_t i = 0; i < n / 2; i++) {
        evens[i] = matrix[2 * i];
        odds[i] = matrix[2 * i + 1];
    }

    vector<complex<double>> e_out = ifft(evens);
    vector<complex<double>> o_out = ifft(odds);

    vector<complex<double>> y(n);
    complex<double> omega(1.0, 0.0);
    //The angle is now positive, and the incorrect division by N is removed.
    complex<double> omega_n = polar(1.0, 2.0 * 3.14 / n);

    for (size_t j = 0; j < n / 2; ++j) {
        complex<double> t = omega * o_out[j];
        y[j] = e_out[j] + t;
        y[j + n / 2] = e_out[j] - t;
        omega *= omega_n;
    }
    return y;
}
//find the closest power of two - returns an int that we use to create a square matrix
//takes in orginial height and width as input
int power_of_two(int n){
    return pow(2, ceil( log2(n) ));
}
//Zero Pads image matrix - takes in the original image matrix, height and width, as well as a padded height and width
//Then it converts the mat into a complex number vector for use in fft;
vector<complex<double>> prepare_matrix(const vector<double>& og_mat, int og_width, int og_height, int padded_height, int padded_width){
    //init a zero matrix
    auto start_time = chrono::high_resolution_clock::now();
    vector<double> mat(padded_width*padded_height, 0);
    //fill up the top left of the matrix with the values
    for(int y = 0; y<og_height; ++y){
        for(int x = 0; x<og_width; ++x){
            mat[y*padded_width+x] = og_mat[y*og_width + x];
        }
    }
    vector<complex<double>> out(mat.begin(), mat.end());
    cout << "Padded Matrix" << endl;
    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    pad_speed += duration.count();
    return out;
}
//takes in a width and height, which will be the size of the original matrix, and a standard deviation to determine blur amount
vector<double> generate_gaussian_function(int width, int height, double std_dev){
    //generates a gaussian kernel centered at 0,0 - the top left of the image
    //This causes the bell curve to wrap around all four corners and prepares it for fft use
    auto start_time = chrono::high_resolution_clock::now();
    vector<double> kernel(width*height);
    double sum = 0.0;
    for(int y = 0; y<height; ++y){
        for(int x = 0; x < width; ++x){
            int x_distance = x > width/2 ? width - x : x;//wrap around logic
            int y_distance = y > height/2 ? height - y : y;
            
            double value = exp(-(x_distance*x_distance + y_distance*y_distance) / (2*std_dev*std_dev));
            sum += value;
            kernel[y*width + x] = value;
        }
    }
    for(int i = 0; i< height*width; ++i){
        kernel[i]/= sum;
    }
    cout << "Kernel Generated" << endl;
    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    gaussian_kernel_gen_speed = duration.count();
    return kernel;

}
//Converts double vector to unsigned vector safely
//takes in the complex double image_matrix, original width and height
vector<unsigned char> back_convert(const vector<complex<double>>& mat, int height, int width, int padded_width){
    auto start_time = chrono::high_resolution_clock::now();
    vector<unsigned char> output;
    output.reserve(height*width);//reserve memory for output with needing to reallocate
    for(int y = 0; y < height; ++y){
        for(int x = 0; x< width; ++x){
            double real = mat[y*padded_width + x].real();
            unsigned char clamped = static_cast<unsigned char>(max(0.0, min(255.0, round(real))));
            output.push_back(clamped);
        }
    }
    cout << "Converted back to real" << endl;
    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    back_convert_speed = duration.count();
    return output;
}
//Complex element multiplication - takes in two complex double vectors and returns on
vector<complex<double>> element_mult(const vector<complex<double>>& mat, const vector<complex<double>>& kernel){
    auto start_time = chrono::high_resolution_clock::now();
    size_t matsize = mat.size();
    vector<complex<double>> out(matsize);
    for(int i = 0; i<matsize; ++i){
        out[i] = mat[i] * kernel[i];
    }
    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    mult_speed = duration.count();
    return out;
}
// Helper function to transpose a matrix
void transpose(vector<complex<double>>& matrix, int width, int height) {
    auto start_time = chrono::high_resolution_clock::now();
    vector<complex<double>> temp(width * height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            temp[x * height + y] = matrix[y * width + x];
        }
    }
    matrix = temp;
    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    transpose_speed += duration.count();
}

//2D FFT function
vector<complex<double>> fft2d(vector<complex<double>>& matrix, int width, int height) {
    auto start_time = chrono::high_resolution_clock::now();
    //FFT on all rows
    for (int y = 0; y < height; ++y) {
        vector<complex<double>> row(matrix.begin() + y * width, matrix.begin() + (y + 1) * width);
        vector<complex<double>> fft_row = fft(row);
        copy(fft_row.begin(), fft_row.end(), matrix.begin() + y * width);
    }

    //Transpose the matrix
    transpose(matrix, width, height);

    //FFT on all new rows (original columns)
    for (int x = 0; x < width; ++x) {
        vector<complex<double>> col(matrix.begin() + x * height, matrix.begin() + (x + 1) * height);
        vector<complex<double>> fft_col = fft(col);
        copy(fft_col.begin(), fft_col.end(), matrix.begin() + x * height);
    }

    //Transpose back to original orientation
    transpose(matrix, height, width);
    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    fft_speed = duration.count();
    return matrix;
}

//2D IFFT function
vector<complex<double>> ifft2d(vector<complex<double>>& matrix, int width, int height) {
    auto start_time = chrono::high_resolution_clock::now();
    //IFFT on all rows
    for (int y = 0; y < height; ++y) {
        vector<complex<double>> row(matrix.begin() + y * width, matrix.begin() + (y + 1) * width);
        vector<complex<double>> ifft_row = ifft(row);
        copy(ifft_row.begin(), ifft_row.end(), matrix.begin() + y * width);
    }

    //Transpose
    transpose(matrix, width, height);

    //IFFT on all new rows (original columns)
    for (int x = 0; x < width; ++x) {
        vector<complex<double>> col(matrix.begin() + x * height, matrix.begin() + (x + 1) * height);
        vector<complex<double>> ifft_col = ifft(col);
        copy(ifft_col.begin(), ifft_col.end(), matrix.begin() + x * height);
    }
    
    //Transpose back
    transpose(matrix, height, width);

    //Normalize the final result by the total number of pixels
    double total_pixels = width * height;
    for (auto& val : matrix) {
        val /= total_pixels;
    }
    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    ifft_speed = duration.count();
    return matrix;
}

//Convolution
//takes in an original grayscale matrix, its dimensions, and a standard deviation for the gaussian filter
vector<unsigned char> gaussian_convolve(const vector<unsigned char>& red, const vector<unsigned char>& green, const vector<unsigned char>& blue, int width, int height, double std_dev){
    auto start_time = chrono::high_resolution_clock::now();  
    vector<unsigned char> og_mat = grayscale_thread_mg(ref(red), green, blue, width, height); 
    auto grayscale_time = chrono::high_resolution_clock::now();
    auto gray_end = chrono::duration_cast<std::chrono::microseconds>(grayscale_time - start_time);
    grayscale_speed = gray_end.count();
    cout << "grayscale_speed: " << grayscale_speed << endl;
    
    //typecast to double vec
    vector<double> image_matrix(og_mat.begin(), og_mat.end());
    //get padded dimensions
    int padded_h = power_of_two(height);
    int padded_w = power_of_two(width);
    //prepare gaussian kernel and zero pad image matrix
    vector<double> gaussian_kernel = generate_gaussian_function(padded_w, padded_h, std_dev);
    vector<complex<double>> normalized_kernel(gaussian_kernel.begin(), gaussian_kernel.end());
    vector<complex<double>> normalized_image_matrix = prepare_matrix(image_matrix, width, height, padded_h, padded_w);
    //run fft on both matrices
    fft2d(normalized_kernel, padded_w, padded_h);
    fft2d(normalized_image_matrix, padded_w, padded_h);
    cout << "Forward 2D FFTS completed" << endl;
    vector<complex<double>> combined = element_mult(normalized_image_matrix, normalized_kernel);
    cout << "Elements Multiplied" << endl;
    //inverse fft
    ifft2d(combined, padded_w, padded_h);
    cout << "Inverse 2D FFT completed" << endl;
    //convert and return output
    vector <unsigned char> final_image = back_convert(combined, height, width, padded_w);

    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    cout << "Timing Breakdown----------------------------" << endl;
    cout << "Function runtime: " << duration.count() << " microseconds" << endl;
    cout << "Grayscale Time: " << grayscale_speed << " microseconds" << endl;
    cout << "Gaussian Kernel Generation Time: " << gaussian_kernel_gen_speed << " microseconds" << endl;
    cout << "Padding Time: " << pad_speed << " microseconds" << endl;
    cout << "FFT Time: " << fft_speed << " microseconds" << endl;
    cout << "Transpose Time: " << transpose_speed << " microseconds" << endl;
    cout << "Element-wise Multiplication Time: " << mult_speed << " microseconds" << endl;
    cout << "IFFT Time: " << ifft_speed << " microseconds" << endl;
    cout << "Back Conversion Time: " << back_convert_speed << " microseconds" << endl;
    return final_image;
}