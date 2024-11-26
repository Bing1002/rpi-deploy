/*
Output: 
shape: 224x224x3
Output 0: -1.19531
Output 1: 0.485107
Output 2: -1.02344
Output 3: -3.22461
Output 4: 0.214355
Output 5: 1.25781
Output 6: -2.33398
Output 7: 1.22168
Output 8: 1.34375
Output 9: -3.38086
Average detection time after 10 iterations: 21.4481 ms
Average detection time after 20 iterations: 21.0212 ms
Average detection time after 30 iterations: 20.8952 ms
Average detection time after 40 iterations: 20.7989 ms
Average detection time after 50 iterations: 20.7131 ms
*/

#include<stdio.h>
#include<iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/imgproc/imgproc.hpp>

#include<ncnn/net.h>
#include<ncnn/mat.h>
#include<ncnn/benchmark.h>


int main(int argc, char** argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [imagepath\n", argv[0]);
        return -1;
    }

    const char* imagepath = argv[1];

    cv::Mat img = cv::imread(imagepath, 1);
    int rows = img.rows;
    int cols = img.cols;
    int channels = img.channels();
    std::cout << "shape: "<< rows << "x" << cols << "x" << channels<<std::endl;

    if (img.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    ncnn::Mat input = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_BGR2RGB, img.cols, img.rows, 224, 224);

    const float mean_vals[3] = {0.485f * 255.0f, 0.456f * 255.0f, 0.406f * 255.0f};
    const float norm_vals[3] = {1 / 0.229f / 255.0f, 1 / 0.224f / 255.0f, 1 / 0.225f / 255.0f};
    input.substract_mean_normalize(mean_vals, norm_vals);

    // print pixels 
    

    // 
    ncnn::Net net;
    net.load_param("resnet18_torchscript.ncnn.param");  // fp16=1
    net.load_model("resnet18_torchscript.ncnn.bin");

    int count = 1000;
    double time_avg = 0;
    for (int i = 0; i < count; i++) {
        double start = ncnn::get_current_time();

        ncnn::Extractor extractor = net.create_extractor();
        extractor.input("in0", input);

        ncnn::Mat output;
        extractor.extract("out0", output);

        if (i == 0) {
                for (int i = 0; i < std::min(10, output.w); i++) {
                    std::cout << "Output " << i << ": " << output[i] << std::endl;
                }
        }

        double end = ncnn::get_current_time();
        double time = end - start;
        time_avg += time;


        if ((i + 1) % 10 == 0) { // Print every 10th iteration
            double avg_time = time_avg / (i + 1);
            std::cout << "Average detection time after " << (i + 1) << " iterations: " << avg_time << " ms" << std::endl;
        }

    }

    // TODO 
    // add softmax + argmax 
    //
    return 0;

}