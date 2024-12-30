
#include <ncnn/net.h>


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <stdlib.h>
#include <float.h>
#include <stdio.h>
#include <vector>

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects)
{
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold, bool agnostic = false)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

            if (!agnostic && a.label != b.label)
                continue;

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

static inline float sigmoid(float x)
{
    return 1.0f / (1.0f + exp(-x));
}

static void generate_proposals(const ncnn::Mat& pred, int stride, const ncnn::Mat& in_pad, float prob_threshold, std::vector<Object>& objects)
{
    const int num_grid = pred.h;

    int num_grid_x = pred.w;
    int num_grid_y = pred.h;

    const int num_class = 80; // number of classes. 80 for COCO
    const int reg_max_1 = (pred.c - num_class) / 4;

    for (int i = 0; i < num_grid_y; i++)
    {
        for (int j = 0; j < num_grid_x; j++)
        {
            // find label with max score
            int label = -1;
            float score = -FLT_MAX;
            for (int k = 0; k < num_class; k++)
            {
                float s = pred.channel(k).row(i)[j];
                if (s > score)
                {
                    label = k;
                    score = s;
                }
            }

            score = sigmoid(score);

            if (score >= prob_threshold)
            {
                ncnn::Mat bbox_pred(reg_max_1, 4);
                for (int k = 0; k < reg_max_1 * 4; k++)
                {
                    bbox_pred[k] = pred.channel(num_class + k).row(i)[j];
                }
                {
                    ncnn::Layer* softmax = ncnn::create_layer("Softmax");

                    ncnn::ParamDict pd;
                    pd.set(0, 1); // axis
                    pd.set(1, 1);
                    softmax->load_param(pd);

                    ncnn::Option opt;
                    opt.num_threads = 1;
                    opt.use_packing_layout = false;

                    softmax->create_pipeline(opt);

                    softmax->forward_inplace(bbox_pred, opt);

                    softmax->destroy_pipeline(opt);

                    delete softmax;
                }

                float pred_ltrb[4];
                for (int k = 0; k < 4; k++)
                {
                    float dis = 0.f;
                    const float* dis_after_sm = bbox_pred.row(k);
                    for (int l = 0; l < reg_max_1; l++)
                    {
                        dis += l * dis_after_sm[l];
                    }

                    pred_ltrb[k] = dis * stride;
                }

                float pb_cx = j * stride;
                float pb_cy = i * stride;

                float x0 = pb_cx - pred_ltrb[0];
                float y0 = pb_cy - pred_ltrb[1];
                float x1 = pb_cx + pred_ltrb[2];
                float y1 = pb_cy + pred_ltrb[3];

                Object obj;
                obj.rect.x = x0;
                obj.rect.y = y0;
                obj.rect.width = x1 - x0;
                obj.rect.height = y1 - y0;
                obj.label = label;
                obj.prob = score;

                objects.push_back(obj);
            }
        }
    }
}

static int detect_nanodet(const cv::Mat& bgr, std::vector<Object>& objects)
{
    ncnn::Net nanodet;

    nanodet.opt.use_vulkan_compute = false;
    // nanodet.opt.use_bf16_storage = true;

    // original pretrained model from https://github.com/RangiLyu/nanodet
    // the ncnn model https://github.com/nihui/ncnn-assets/tree/master/models
    //     nanodet.load_param("nanodet-plus-m_320.torchscript.ncnn.param");
    //     nanodet.load_model("nanodet-plus-m_320.torchscript.ncnn.bin");
    if (nanodet.load_param("/home/bing/code/checkpoints/nanodet/nanodet_plus_m_416.torchscript.ncnn.param"))
        exit(-1);
    if (nanodet.load_model("/home/bing/code/checkpoints/nanodet/nanodet_plus_m_416.torchscript.ncnn.bin"))
        exit(-1);

    int width = bgr.cols;
    int height = bgr.rows;

    //     const int target_size = 320;
    const int target_size = 416;
    const float prob_threshold = 0.4f;
    const float nms_threshold = 0.5f;

    // pad to multiple of 32
    int w = target_size;
    int h = target_size;
    float scale_w = (float)width / w;
    float scale_h = (float)height / h;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, width, height, w, h);
    fprintf(stderr, "w: %d, h: %d, c: %d\n", in.w, in.h, in.c);
    for (int c = 0; c < in.c; c++)
    {
        float* ptr = in.channel(c);
        for (int i = 0; i < 6; i++)
        {
            fprintf(stderr, "pixel %d: %f\n", i, ptr[i]);
        }
    }

    // pad to target_size rectangle
    int wpad = (w + 31) / 32 * 32 - w;
    int hpad = (h + 31) / 32 * 32 - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 0.f);
    
    // normalized_pixel = (original_pixel - mean) * norm 
    // norm = (1/255, 1/255, 1/255) or (1/57.375, 1/57.12, 1/58.395)

    const float mean_vals[3] = {103.53f, 116.28f, 123.675f};
    const float norm_vals[3] = {0.017429f, 0.017507f, 0.017125f};
    in_pad.substract_mean_normalize(mean_vals, norm_vals);
    for (int c = 0; c < in.c; c++)
    {
        float* ptr = in_pad.channel(c);
        for (int i = 0; i < 6; i++)
        {
            fprintf(stderr, "pixel %d: %f\n", i, ptr[i]);
        }
    }

    ncnn::Extractor ex = nanodet.create_extractor();
    fprintf(stderr, "in_pad.w: %d, in_pad.h: %d\n", in_pad.w, in_pad.h);
    ex.input("in0", in_pad);


    ncnn::Mat out;
    ex.extract("out0", out);
    fprintf(stderr, "out.w: %d, out.h: %d, out.c: %d\n", out.w, out.h, out.c);

    //
    fprintf(stderr, "width: %d, height: %d, channels: %d, dims: %d\n", out.w, out.h, out.c, out.dims);
    fprintf(stderr, "total size: %d\n", out.total());
    fprintf(stderr, "cstep size: %d\n", out.cstep);
    fprintf(stderr, "elemsize: %d, elempack: %d\n", out.elemsize, out.elempack);

    // channel1: row1 (col1, col2, ...), row2, row3
    // channel2: row1, row2, row3
    // ...
    // 
    for (int q = 0; q < out.c; q++)
    {
        float* ptr = out.channel(q);
        for (int z = 0; z < out.h; z++)  // row 
        {
            for (int y = 0; y < out.w; y++)  // col 
            {
                printf("%f ", ptr[y]);
            }
            ptr += out.w;
            printf("\n");
            break;
        }
        printf("\n");
        break;
    }


    std::vector<Object> proposals;

    int strides[] = { 8, 16, 32, 64 }; // strides of the multi-level feature.
    int num_strides = sizeof(strides) / sizeof(strides[0]);
    int start_index = 0;

    for (int i = 0; i < num_strides; i++)
    {
        int size = pow((target_size / strides[i]), 2);
        // ncnn::Mat part1 = out.channel_range(0, 2704);
        // ncnn::Mat part2 = out.channel_range(2704, 676);
        // ncnn::Mat part3 = out.channel_range(3380, 169);
        // ncnn::Mat part4 = out.channel_range(3549, 49);
        ncnn::Mat pred = out.range(start_index, size);
        start_index += size;

        std::vector<Object> objects_i;
        generate_proposals(pred, strides[i], in_pad, prob_threshold, objects_i);

        proposals.insert(proposals.end(), objects_i.begin(), objects_i.end());
    }

    
    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    int count = picked.size();

    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (objects[i].rect.x - (wpad / 2)) / scale_w;
        float y0 = (objects[i].rect.y - (hpad / 2)) / scale_h;
        float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale_w;
        float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale_h;

        // clip
        x0 = std::max(std::min(x0, (float)(width - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(height - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(width - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(height - 1)), 0.f);

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

    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::rectangle(image, obj.rect, cv::Scalar(255, 0, 0));

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

    cv::imshow("image", image);
    cv::waitKey(0);
}



int main(int argc, char** argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }

    const char* imagepath = argv[1];

    cv::Mat m = cv::imread(imagepath, 1);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }
    fprintf(stderr, "w: %d, h: %d, c: %d\n", m.cols, m.rows, m.channels());
    fprintf(stderr, "step 0: %d, step 1: %d, elemSize: %d, totoal: %d\n", m.step[0], m.step[1], m.elemSize(), m.total());

    // opencv matrix to ncnn mat

    for (int row = 0; row < m.rows; row++) {
        for (int col = 0; col < m.cols; col++) {
            for (int channel = 0; channel < m.channels(); channel++) {
                // BGR order 
                // fprintf(stderr, "pixel %d: %d, %d, %d\n", row, col, channel, m.at<cv::Vec3b>(row, col)[channel]);
                // fprintf(stderr, "pixel %d: %d, %d, %d\n", row, col, channel, (int)(*(m.data + m.step[0] * row + m.step[1] * col + channel)));
                
            }
        }
    }


    std::vector<Object> objects;
    detect_nanodet(m, objects);

    // draw_objects(m, objects);

    return 0;
}
