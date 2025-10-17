#include <CL/cl.h>           // Add this line first
#include <CL/opencl.hpp>
#include "image_ops.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <string>
#include <complex>
#include <chrono>
#include <thread>
#include <functional>
#include <execution>
#include <immintrin.h>
using namespace std;


//timing var speeds
long long int grayscale_speed = 0;//
long long int fft_speed = 0;//
long long int gaussian_kernel_gen_speed = 0;//
long long int pad_speed = 0;//
long long transpose_speed = 0;//
long long back_convert_speed = 0;//
long long int ifft_speed = 0;//
long long int mult_speed = 0;//

//returns a simple grayscale image by averaging the three channels,
//takes in the three channels, and width and height as input
// OpenCL kernel - executed on the GPU
const char *kernelSource = R"(
__kernel void graysc(__global const unsigned char *r, 
                         __global const unsigned char *g,
                         __global const unsigned char *b, 
                         __global unsigned char *c, 
                         const unsigned int n) {
    int id = get_global_id(0);
    if (id < n) {
        c[id] = (r[id] + g[id] + b[id])/3;
    }
}
__kernel void normalize(__global const unsigned char *gray, __global double *real,
                        const unsigned int width,
                        const unsigned int pad_w,
                        const unsigned int height){
    int x_idx = get_global_id(0);
    int y_idx = get_global_id(1);
    if (x_idx < width && y_idx < height) {
        real[y_idx*pad_w + x_idx] = (double)gray[y_idx*width + x_idx];      
    }              
}    
inline double2 c_mult(double2 a, double2 b) {
    return (double2)(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__kernel void bit_reverse_2d(__global const double *in_real, 
                           __global const double *in_imag,
                           __global double *out_real, 
                           __global double *out_imag,
                           const int n,
                           const int is_row_pass) { // 1 for row, 0 for col
    int x = get_global_id(0);
    int y = get_global_id(1);
    int width = get_global_size(0);
    int height = get_global_size(1);

    int i = is_row_pass ? x : y;
    if (i >= n) return;

    int reversed_i = 0;
    int temp_i = i;
    int log2_n = (int)(log2((float)n));

    for (int j = 0; j < log2_n; j++) {
        reversed_i <<= 1;
        reversed_i |= (temp_i & 1);
        temp_i >>= 1;
    }
    
    int in_idx  = is_row_pass ? (y * width + reversed_i) : (reversed_i * width + x);
    int out_idx = is_row_pass ? (y * width + x)         : (y * width + x);

    out_real[out_idx] = in_real[in_idx];
    out_imag[out_idx] = in_imag[in_idx];
}


__kernel void fft_1d_pass_2d(__global double *real, 
                           __global double *imag,
                           const int n,
                           const int is_row_pass, // 1 for row, 0 for col
                           const int direction,
                           const int s) {

    int x = get_global_id(0);
    int y = get_global_id(1);
    int width = get_global_size(0);
    int height = get_global_size(1);
    
    double PI = 3.14159265358979323846;
    
    int i = is_row_pass ? x : y;
    if (i >= n) return;
    
    int m = 1 << s;
    int m_half = m >> 1;

    if ((i % m) < m_half) {
        int stride = is_row_pass ? 1 : width;
        int offset = is_row_pass ? (y * width) : x;

        int even_idx = offset + i * stride;
        int odd_idx  = offset + (i + m_half) * stride;

        double2 even = (double2)(real[even_idx], imag[even_idx]);
        double2 odd  = (double2)(real[odd_idx],  imag[odd_idx]);

        int butterfly_idx = i % m_half;
        double angle = direction * -2.0 * PI * butterfly_idx / m;
        double2 omega = (double2)(cos(angle), sin(angle));
        
        double2 t = c_mult(omega, odd);
        
        real[even_idx] = even.x + t.x;
        imag[even_idx] = even.y + t.y;

        real[odd_idx] = even.x - t.x;
        imag[odd_idx] = even.y - t.y;
    }
}
__kernel void elem_mult(__global double *gray_real, __global double *gray_img, 
                        __global double *gauss_real, __global double *gauss_img,
                        int n) {   
    int id = get_global_id(0);
    if(id < n){
        double temp_real = gray_real[id];
        gray_real[id] = temp_real * gauss_real[id] - gray_img[id] * gauss_img[id];
        gray_img[id] = temp_real * gauss_img[id] + gray_img[id] * gauss_real[id];
    }   
}
__kernel void ifft_normalize(__global double *real, 
                             __global double *imag,
                             const int n) {
    int id = get_global_id(0);
    double total_pixels = (double)n;
    if (id < n) {
        real[id] /= total_pixels;
        imag[id] /= total_pixels;
    }
}
__kernel void convert(__global double *gray_real, __global unsigned char *g_out, int height, int width, int pad_w){
    int y_id = get_global_id(1);
    int x_id = get_global_id(0);
    if(y_id < height && x_id < width){
        double real = max(0.0, min(255.0, round(gray_real[y_id*pad_w + x_id])));
        g_out[y_id * width + x_id] = (unsigned char)real;
    }

}                             
)";

//find the closest power of two - returns an int that we use to create a square matrix
//takes in orginial height and width as input
int power_of_two(int n){
    int p = 1;
    while(p < n){
        p <<= 1;
    }
    return p;
}
//takes in a width and height, which will be the size of the original matrix, and a standard deviation to determine blur amount
void generate_gaussian_function(vector<double>& kernel, int width, int height, double std_dev){
    //generates a gaussian kernel centered at 0,0 - the top left of the image
    //This causes the bell curve to wrap around all four corners and prepares it for fft use
    auto start_time = chrono::high_resolution_clock::now();
    //Multithreaded approach
    int num_threads = thread::hardware_concurrency() - 2;
    if(num_threads == 0)num_threads = 1;
    cout << "threads: " << num_threads << endl;
    int block_size = (height) / num_threads;
    vector<thread> threads;
    vector<double> sums(num_threads, 0.0);
    //pre compute distances to save lots of iteration
    //Intrinsics, AMD Ryzen AI 7PRO has 4 512 bit vector ALU's
    //Double Bit precision is 64 bits so we can use 8 of these in a set
    double two_sigma = 2*std_dev*std_dev;
    for(int i = 0; i< num_threads; i++){
        int start_y = i*block_size;
        int end_y = (i == num_threads - 1) ? height : (i+1)*block_size;
        threads.emplace_back([&sums, &kernel, width, height, start_y, end_y, std_dev, i, two_sigma]() {
            double sum = 0.0;
            for(int y = start_y; y<end_y; ++y){
                for(int x = 0; x < width; ++x){
                    int x_distance = min(x, width -x);//wrap around logic
                    int y_distance = min(y, height - y);
                    double value = exp(-(x_distance*x_distance + y_distance*y_distance) / (two_sigma));
                    sum += value;
                    kernel[y*width + x] = value;
                }
            }
            sums[i] = sum;
        });
    }
    for (std::thread& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }
    threads.clear();

    //Do the division
    double sum = accumulate(sums.begin(), sums.end(), 0.0);
    block_size = (width*height) / num_threads;
    for(int i = 0; i< num_threads; i++){
        int start = i*block_size;
        int end = (i == num_threads - 1) ? (width*height) : (i+1)*block_size;
        
        threads.emplace_back([&kernel, sum, start, end]() {
            __m512d sum_vec = _mm512_set1_pd(sum);
            for(int j = start; j<end; j+=8){
                //run some simd intrinsics
                if(j+8 < end){
                    __m512d kernel_vec = _mm512_loadu_pd(&kernel[j]);
                    __m512d value = _mm512_div_pd(kernel_vec, sum_vec);
                    double temp [8];
                    _mm512_storeu_pd(temp, value);
                    for (int k = 0; k < 8; ++k) {
                        kernel[j+k] = temp[k];
                    }
                }else{
                    for(int l = j; l < end; l++){
                        kernel[l] /= sum;
                    }
                }

            }        
        });
    }
    for (std::thread& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }
   
    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    cout << "Kernel Generated: " << duration.count() << endl;
}
void fft_2d(cl::CommandQueue& queue, cl::Kernel& bit_reverse_kernel, cl::Kernel& fft_pass_kernel,
            cl::Buffer& d_real, cl::Buffer& d_imag, cl::Buffer& d_temp_real, cl::Buffer& d_temp_imag,
            int width, int height, int direction) {

    cl::NDRange global_work_size(width, height);

    // --- Row Pass ---
    bit_reverse_kernel.setArg(0, d_real);
    bit_reverse_kernel.setArg(1, d_imag);
    bit_reverse_kernel.setArg(2, d_temp_real);
    bit_reverse_kernel.setArg(3, d_temp_imag);
    bit_reverse_kernel.setArg(4, width);
    bit_reverse_kernel.setArg(5, 1); // is_row_pass = true
    queue.enqueueNDRangeKernel(bit_reverse_kernel, cl::NullRange, global_work_size, cl::NullRange);

    fft_pass_kernel.setArg(0, d_temp_real);
    fft_pass_kernel.setArg(1, d_temp_imag);
    fft_pass_kernel.setArg(2, width);
    fft_pass_kernel.setArg(3, 1); // is_row_pass = true
    fft_pass_kernel.setArg(4, direction);
    int log2_w = static_cast<int>(log2(width));
    for (int s = 1; s <= log2_w; ++s) {
        fft_pass_kernel.setArg(5, s);
        queue.enqueueNDRangeKernel(fft_pass_kernel, cl::NullRange, global_work_size, cl::NullRange);
    }
    
    // --- Column Pass ---
    bit_reverse_kernel.setArg(0, d_temp_real);
    bit_reverse_kernel.setArg(1, d_temp_imag);
    bit_reverse_kernel.setArg(2, d_real);
    bit_reverse_kernel.setArg(3, d_imag);
    bit_reverse_kernel.setArg(4, height);
    bit_reverse_kernel.setArg(5, 0); // is_row_pass = false
    queue.enqueueNDRangeKernel(bit_reverse_kernel, cl::NullRange, global_work_size, cl::NullRange);

    fft_pass_kernel.setArg(0, d_real);
    fft_pass_kernel.setArg(1, d_imag);
    fft_pass_kernel.setArg(2, height);
    fft_pass_kernel.setArg(3, 0); // is_row_pass = false
    fft_pass_kernel.setArg(4, direction);
    int log2_h = static_cast<int>(log2(height));
    for (int s = 1; s <= log2_h; ++s) {
        fft_pass_kernel.setArg(5, s);
        queue.enqueueNDRangeKernel(fft_pass_kernel, cl::NullRange, global_work_size, cl::NullRange);
    }
    queue.finish();
}
//Convolution
//takes in an original grayscale matrix, its dimensions, and a standard deviation for the gaussian filter
vector<unsigned char> gaussian_convolve(const vector<unsigned char>& red, const vector<unsigned char>& green, const vector<unsigned char>& blue, int width, int height, double std_dev){
    auto start = chrono::high_resolution_clock::now();
    int N = width*height;
    int pad_h = power_of_two(height);
    int pad_w = power_of_two(width);
    //launch threaded gaussian_kernel_generation
    vector<double> gaussian_kernel(pad_h*pad_w);
    thread gauss_gen([&gaussian_kernel, pad_w, pad_h, std_dev ](){
        generate_gaussian_function(gaussian_kernel, pad_w, pad_h, std_dev);
    });

    //Set Up OpenCL
    vector<unsigned char> out(N);
    vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty()) {
        cerr << "No OpenCL platforms found\n";
    }
    cl::Platform platform = platforms[0];
    cout << "Using platform: " << platform.getInfo<CL_PLATFORM_NAME>() << endl;
    //Get all GPU devices
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    if (devices.empty()) {
        cerr << "No GPU devices found" << endl;
    }
    cl::Device device = devices[0];
    cout << "Using device: " << device.getInfo<CL_DEVICE_NAME>() << endl;
    //Create context and command queue
    cl::Context context(device);
    cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);
    //Create and build program
    cl::Program::Sources sources;
    sources.push_back({kernelSource, strlen(kernelSource)});
    cl::Program program(context, sources);
    program.build({device});
    
    //Create kernel
    cl::Kernel kernel(program, "graysc");
    
    //Create device buffers
    cl::Buffer d_r(context, CL_MEM_READ_ONLY, sizeof(unsigned char) * N);
    cl::Buffer d_g(context, CL_MEM_READ_ONLY, sizeof(unsigned char) * N);
    cl::Buffer d_b(context, CL_MEM_READ_ONLY, sizeof(unsigned char) * N);
    cl::Buffer d_o(context, CL_MEM_WRITE_ONLY, sizeof(unsigned char) * N);
    auto setup_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<std::chrono::microseconds>(setup_time - start);
    cout << "Setup Time: " << duration.count() << endl; 

    
    //Copy data to device
    auto start_time = chrono::high_resolution_clock::now();
    queue.enqueueWriteBuffer(d_r, CL_TRUE, 0, sizeof(unsigned char) * N, red.data());
    queue.enqueueWriteBuffer(d_g, CL_TRUE, 0, sizeof(unsigned char) * N, green.data());
    queue.enqueueWriteBuffer(d_b, CL_TRUE, 0, sizeof(unsigned char) * N, blue.data());
    auto end_time = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    cout << "Upload Time: " << duration.count() << endl; 

    //Set kernel arguments
    kernel.setArg(0, d_r);
    kernel.setArg(1, d_g);
    kernel.setArg(2, d_b);
    kernel.setArg(3, d_o);
    kernel.setArg(4, N);
    //Execute kernel
    cl::NDRange globalSize(N);
    start_time = chrono::high_resolution_clock::now();
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, cl::NullRange);
    queue.finish();
    end_time = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    cout << "Run Time - Grayscale: " << duration.count() << endl; 
    
    //join gaussian thread--------------------------------------------------------------------
    gauss_gen.join();
    start_time = chrono::high_resolution_clock::now();
    //setup normalize grayscale
    cl::Kernel normalize_kernel(program, "normalize");
    //Create device buffers -- going to correspond to the real and imaginary components of the gaussian and grayscale mat
    int padded_N = pad_h*pad_w;
    vector<double> real_gray(padded_N, 0.0);
    vector<double> im_gray(padded_N, 0.0);
    vector<double> im_gauss(padded_N, 0.0);
    cl::Buffer d_real_gray(context, CL_MEM_READ_WRITE, sizeof(double) * padded_N);
    cl::Buffer d_im_gray(context, CL_MEM_READ_WRITE, sizeof(double) * padded_N);
    cl::Buffer d_real_gauss(context, CL_MEM_READ_WRITE, sizeof(double) * padded_N);
    cl::Buffer d_im_gauss(context, CL_MEM_READ_WRITE, sizeof(double) * padded_N);
    cl::Buffer d_temp_real(context, CL_MEM_READ_WRITE, sizeof(double) * padded_N);
    cl::Buffer d_temp_imag(context, CL_MEM_READ_WRITE, sizeof(double) * padded_N);

    //copy gauss over -- last time you have to set this up
    queue.enqueueWriteBuffer(d_real_gauss, CL_TRUE, 0, sizeof(double) * padded_N, gaussian_kernel.data());
    queue.enqueueWriteBuffer(d_im_gauss, CL_TRUE, 0, sizeof(double) * padded_N, im_gauss.data());
    queue.enqueueWriteBuffer(d_real_gray, CL_TRUE, 0, sizeof(double) * padded_N, real_gray.data());
    queue.enqueueWriteBuffer(d_im_gauss, CL_TRUE, 0, sizeof(double) * padded_N, im_gray.data());
    end_time = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    //execute kernel
    normalize_kernel.setArg(0, d_o);//use d_o from last time
    normalize_kernel.setArg(1, d_real_gray);
    normalize_kernel.setArg(2, width);
    normalize_kernel.setArg(3, pad_w);
    normalize_kernel.setArg(4, height);
    start_time = chrono::high_resolution_clock::now();
    cl::NDRange globalSizePadded(pad_w, pad_h);
    queue.enqueueNDRangeKernel(normalize_kernel, cl::NullRange, globalSizePadded, cl::NullRange);
    queue.finish();
    end_time = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    cout << "Run Time - Pad: " << duration.count() << endl; 
    //-----------------------------FFT Implementations----------------------------------------------------------
    auto fft_start_time = chrono::high_resolution_clock::now();
    cl::Kernel bit_reverse_kernel(program, "bit_reverse_2d");
    cl::Kernel fft_pass_kernel(program, "fft_1d_pass_2d");
    cl::NDRange global_work_size(pad_w, pad_h);
    
    // --- FFT Gray Image ---
    fft_2d(queue, bit_reverse_kernel, fft_pass_kernel, d_real_gray, d_im_gray, d_temp_real, d_temp_imag, pad_w, pad_h, 1);
    
    // --- FFT Gaussian Kernel ---
    fft_2d(queue, bit_reverse_kernel, fft_pass_kernel, d_real_gauss, d_im_gauss, d_temp_real, d_temp_imag, pad_w, pad_h, 1);

    auto fft_end_time = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<std::chrono::microseconds>(fft_end_time - fft_start_time);
    cout << "Run Time - All FFTs: " << duration.count() << endl;

    // --- Element Multiplication ---
    start_time = chrono::high_resolution_clock::now();
    cl::Kernel elem_mult_kernel(program, "elem_mult");
    elem_mult_kernel.setArg(0, d_real_gray);
    elem_mult_kernel.setArg(1, d_im_gray);
    elem_mult_kernel.setArg(2, d_real_gauss);
    elem_mult_kernel.setArg(3, d_im_gauss);
    elem_mult_kernel.setArg(4, padded_N);
    queue.enqueueNDRangeKernel(elem_mult_kernel, cl::NullRange, cl::NDRange(padded_N), cl::NullRange);
    queue.finish();
    end_time = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    cout << "Run Time - Element Multiplication: " << duration.count() << endl; 

    // --- IFFT the Result ---
    start_time = chrono::high_resolution_clock::now();
    fft_2d(queue, bit_reverse_kernel, fft_pass_kernel, d_real_gray, d_im_gray, d_temp_real, d_temp_imag, pad_w, pad_h, -1);
    
    // IFFT Normalization
    cl::Kernel normalize_ifft_kernel(program, "ifft_normalize");
    normalize_ifft_kernel.setArg(0, d_real_gray);
    normalize_ifft_kernel.setArg(1, d_im_gray);
    normalize_ifft_kernel.setArg(2, padded_N);
    queue.enqueueNDRangeKernel(normalize_ifft_kernel, cl::NullRange, cl::NDRange(padded_N), cl::NullRange);
    queue.finish();
    end_time = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    cout << "Run Time - IFFT & Normalize: " << duration.count()<< endl;
    //back-convert
    start_time = chrono::high_resolution_clock::now();
    cl::Kernel convert_kernel(program, "convert");
    convert_kernel.setArg(0, d_real_gray);
    convert_kernel.setArg(1, d_o);
    convert_kernel.setArg(2, height);
    convert_kernel.setArg(3, width);
    convert_kernel.setArg(4, pad_w);
    queue.enqueueNDRangeKernel(convert_kernel, cl::NullRange, cl::NDRange(width, height), cl::NullRange);
    queue.finish();
    queue.enqueueReadBuffer(d_o, CL_TRUE, 0, sizeof(unsigned char) * N, out.data());
    queue.finish();
    end_time = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    cout << "Run Time - convert: " << duration.count() << endl;

    duration = chrono::duration_cast<std::chrono::microseconds>(end_time - start);

    cout << "Timing Breakdown----------------------------" << endl;
    cout << "Function runtime: " << duration.count() << " microseconds" << endl;
    
    return out;
}