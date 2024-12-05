
# Object detection model training 
Please refer to this repo: https://github.com/jahongir7174/YOLOv8-pt. We choose the yolov8 nano [model](https://github.com/jahongir7174/YOLOv8-pt/blob/master/weights/best.pt)

Train (DDP):
- Time: 4 GPU, 32 BS, ~4Min/epoch

Test: 
- mAP: 0.368

# Object detection model inference using python and c++

## Python 
```
./python/yolov8n.ipynb
```


## C++
- data: [sample video 1](https://drive.google.com/file/d/1XaYWKBBXBTEo48LR1bzaA1kKkpmIFlEQ/view?usp=drive_link), [sample video 2](https://drive.google.com/file/d/1aYERwBX9SmDQsl4Pibqp2270MSZu0ZDz/view?usp=drive_link)
- compile & run code 
```
cd ./cpp/yolov8n
mdkir build && cd build 
cmake ..
make 
./main 
```
