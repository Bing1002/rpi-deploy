# A resnet18 model inference using python and c++


## Python inference 
- IR - pytorch raw model 

```
./python/resnet18.ipynb
```

## C++ inference 
- IR - pnnx 
- Inference engine - ncnn 

```c++
./resnet18 [image_path]
```



## Python training (DDP)
train resnet18 from scratch on cifar10 dataset using DDP (data distributed parallelism), best accuracy on testing dataset can reach to $94\%$. 

```python
./python/train.py  # single GPU

torchrun --standalone --nproc_per_node=4 train.py  # multiple GPUs
```

