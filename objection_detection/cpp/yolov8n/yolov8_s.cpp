
#include<cstdio>

#include <opencv2/opencv.hpp>

#include<ncnn/net.h>


#define MAX_STRIDE 32

struct Object {
    cv::Rect_<float> rect;
    int label;
    float prob;
};

static inline float intersection_over_union(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static int partition(std::vector<Object>& objects, int left, int right) {
    float pivot = objects[(left + right) / 2].prob; 
    int i = left - 1;

    for (int j = left; j < right; j++) {
        if (objects[j].prob > pivot) {
            i++;
            std::swap(objects[i], objects[j]);
        }
    }
    std::swap(objects[i+1], objects[right]);
    return i + 1;
}


static void qsort_descent_inplace(std::vector<Object>& objects, int left, int right)
{
    if (left < right) {
        int pivot_idx = partition(objects, left, right);

        #pragma omp parallel sections 
        {
            #pragma omp section 
            {
                qsort_descent_inplace(objects, left, pivot_idx - 1);
            }
            #pragma omp section 
            {
                qsort_descent_inplace(objects, pivot_idx + 1, right);
            }
        }

    }

}


static void qsort_descent_inplace(std::vector<Object>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() -1);
}


static void nms_sorted_bboxes(const std::vector<Object>& objects, std::vector<int>& picked, float nms_threshold, bool agnostic = false)
{
    picked.clear();
    const int n = objects.size();
    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = objects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = objects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = objects[picked[j]];
            if (!agnostic && a.label != b.label)
                continue;

            // iou 
            float inter_area = intersection_over_union(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;

            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}


static inline float sigmoid(float x)
{
    return static_cast<float>(1.f / (1.f + exp(-x)));
}

static inline float clampf(float d, float min, float max)
{
    const float t = d < min ? min : d;
    return t > max ? max : t;
}

static void parse_yolov8_detections(
    float* inputs, float confidence_threshold, 
    int num_channels, int num_anchors, int num_labels, 
    int infer_img_width, int infer_img_height, 
    std::vector<Object>& objects
)
{
    std::vector<Object> detections;
    cv::Mat output = cv::Mat(num_channels, num_anchors, CV_32F, inputs).t();

    for (int i = 0; i < num_anchors; i++)
    {
        const float* row_ptr = output.row(i).ptr<float>();
        const float* bboxes_ptr = row_ptr;
        const float* scores_ptr = row_ptr + 4;
        const float* max_s_ptr = std::max_element(scores_ptr, scores_ptr + num_labels);
        float score = *max_s_ptr;
        if (score > confidence_threshold)
        {
            float x = *bboxes_ptr++;
            float y = *bboxes_ptr++;
            float w = *bboxes_ptr++;
            float h = *bboxes_ptr;
            
            float x0 = clampf((x - 0.5f * w), 0.f, (float)infer_img_width);
            float y0 = clampf((y - 0.5f * h), 0.f, (float)infer_img_height);
            float x1 = clampf((x + 0.5f * w), 0.f, (float)infer_img_width);
            float y1 = clampf((y + 0.5f * h), 0.f, (float)infer_img_height);
            
            cv::Rect_<float> bbox;
            bbox.x = x0;
            bbox.y = y0;
            bbox.width = x1 - x0;
            bbox.height = y1 - y0;
            Object object;
            object.label = max_s_ptr - scores_ptr;
            object.prob = score;
            object.rect = bbox;
            detections.push_back(object);
        }
    }
    objects = detections;
}


static int detect_yolov8(const cv::Mat &bgr, std::vector<Object>& objects)
{
    ncnn::Net yolov8;
    yolov8.load_param("/home/bing/code/checkpoints/yolov8n/yolov8n_torchscript.ncnn.param");
    yolov8.load_model("/home/bing/code/checkpoints/yolov8n/yolov8n_torchscript.ncnn.bin");

    const int target_size = 640;
    const float prob_threshold = 0.45f;
    const float nums_threshold = 0.65f;

    int img_w = bgr.cols;
    int img_h = bgr.rows;

    // letterbox pad to multiple of MAX_STRIDE
    int w = img_w;
    int h = img_h;
    float scale = 1.f;

    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }

    // cv::Mat --> ncnn::Mat
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, w, h);
    int wpad = (target_size + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - w;
    int hpad = (target_size + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad /2, ncnn::BORDER_CONSTANT, 114.f);

    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in_pad.substract_mean_normalize(0, norm_vals);

    ncnn::Extractor ex = yolov8.create_extractor();
    ex.input("in0", in_pad);

    std::vector<Object> proposals;

    // stride 32
    {
        ncnn::Mat out; 
        ex.extract("out0", out);
        
        std::vector<Object> objects32;
        const int num_labels = 80;  // COCO has detect 80 object labels
        parse_yolov8_detections(
            (float*)out.data, prob_threshold, 
            out.h, out.w, num_labels, 
            in_pad.w, in_pad.h, 
            objects32
        );
        proposals.insert(proposals.end(), objects32.begin(), objects32.end());
    }

    // sort all proposals by score from highest to lowest 
    qsort_descent_inplace(proposals);

    // nms 
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nums_threshold);

    int count = picked.size();
    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded 
        float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
        float y0 = (objects[i].rect.y - (hpad / 2)) / scale; 
        float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;
        
        // clip 
        // [0, width-1]
        x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;

    }

    return 0;
}


static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects) 
{
    static const char* class_names[] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    };

    static const unsigned char colors[20][3] = {
        {255, 0, 0},    // Red
        {0, 255, 0},    // Green
        {0, 0, 255},    // Blue
        {255, 255, 0},  // Yellow
        {255, 0, 255},  // Magenta
        {0, 255, 255},  // Cyan
        {192, 192, 192}, // Silver
        {128, 128, 128}, // Gray
        {128, 0, 0},     // Maroon
        {128, 128, 0},   // Olive
        {0, 128, 0},     // Dark Green
        {128, 0, 128},   // Purple
        {0, 0, 128},     // Navy
        {255, 165, 0},   // Orange
        {75, 0, 130},    // Indigo
        {240, 230, 140}, // Khaki
        {255, 20, 147},   // Deep Pink
        {255, 228, 196}, // Bisque
        {135, 206, 235}, // Sky Blue
        {240, 128, 128}   // Light Coral
    };

    int color_index = 0;
    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        const unsigned char* color = colors[color_index % 20];
        color_index++;

        cv::Scalar cc(color[0], color[1], color[2]);
        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob, 
        obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        // bbox 
        cv::rectangle(image, obj.rect, cc, 2);

        // caption 
        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseline = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseline;
        if (y<0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;
        
        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseline)),
                      cc, -1);
        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));

    }
    cv::imwrite("result.png", image);


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
        printf("mat is contiguous.\n");
    }

    std::vector<Object> objects;
    detect_yolov8(m, objects);

    draw_objects(m, objects);

    return 0;

}