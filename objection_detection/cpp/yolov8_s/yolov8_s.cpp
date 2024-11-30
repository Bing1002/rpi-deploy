
#include<cstdio>

#include <opencv2/opencv.hpp>


struct Object {
    cv::Rect_<float> rect;
    int label;
    float prob;
};


static int detect_yolov8(const cv::Mat &bgr, std::vector<Object>& objects)
{
    return 0;
}


int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }

    const char* imagepath = argv[1];

    // opencv load image
    cv::Mat m = cv::imread(imagepath, 1);
    if (m.empty()) {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }
    if (m.isContinuous()) {
        printf("mat is contiguous");
    }

    std::vector<Object> objects;
    detect_yolov8(m, objects);



    return 0;

}