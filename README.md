
# Deploy NN models on Raspberry Pi 5 
This hobby project is a self-learning initiative aimed at understanding the deployment of neural network models on edge devices, such as the Raspberry Pi 5, starting from the basics.



## Introduction 



## Tasks 
- image classification 
- object detection 
- object tracking (SOT, MOT)
- instance segmentation 
- pose estimation 
- clip 
- llm (optional)
- vlm (optional)

## Hardware 
CPU: Arm Cortext-A76 CPU, 2.4GHz * 4, Neon  
GPU: VideoCore VII (integrated graph cards), Vulkan 1.3    
Memory: 8Gb



## Measure/Benchmark 
no batch, no dynamic shape

| Hardware | Model | Input Resolution | Batch | Data type | Sparsity | Params | GFLOPs/MACs | Accuracy | FPS | Latency (ms) |Energy | Cost ($) | Comments |
|---------------|---------------|------------------|---------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|
| RPI 5 @4 thread  | resnet18 | 3x224x224  | 1  | fp32 | 0  | 11.689512  | 1.81  | N/A  | N/A  | 20  | N/A  | N/A  |N/A  |
| RPI 5 @4 thread  | yolov8_n | 3x640x640  | 1  | fp32 | 0 | Row | Row | Row | ~9  | 115  |Row |Row |Row |
| RPI 5 @4 thread  | yolov8_n | 3x640x640  | 1  | int8 | 0 | Row | Row | Row | ~9  | 115  |Row |Row |Row |
| RPI 5 @4 thread  | yolox_s | Row | Row |Row |Row |Row |Row |Row |Row |Row |Row |Row | Row |
| RPI 5 @4 thread  | fcos | Row | Row |Row |Row |Row |Row |Row |Row |Row |Row |Row |Row |
| RPI 5 @4 thread  | bytetrack | Row | Row |Row |Row |Row |Row |Row |Row |Row |Row |Row |Row |
| RPI 5 @4 thread  | rtmpose | Row | Row |Row |Row |Row |Row |Row |Row |Row |Row |Row |Row |
| RPI 5 @4 thread  | clip | Row | Row |Row |Row |Row |Row |Row |Row |Row |Row |Row |Row |
| RPI 5 @4 thread  | llm | Row | Row |Row |Row |Row |Row |Row |Row |Row |Row |Row |Row |
| RPI 5 @4 thread  | vlm | Row | Row |Row |Row |Row |Row |Row |Row |Row |Row |Row |Row |



## Deployment framework 

### NCNN
- compile ncnn 
```cmake
cmake -DCMAKE_BUILD_TYPE=Release -DNCNN_VULKAN=OFF -DNCNN_BUILD_EXAMPLES=ON -DNCNN_BUILD_BENCHMARK=ON -DNCNN_BENCHMARK=OFF ..
```
- compile pcnn 



### Model format choices 
- pytorch checkpoint 
- torchscript (scripting vs. tracing)
- ONNX 
- PNNX  


### Model conversion 
- raw model --> onnx IR --> ncnn (deprecated)
- raw model --> torchscript IR (tracing) --> pnnx IR --> ncnn

### Model compression 
- quantization
    - PTQ
    - QAT
- pruning 
- knowledge distillation 
- low-rank (optional) 
- nas (optional)


### TODO 