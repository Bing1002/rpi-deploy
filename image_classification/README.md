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


### Typical phenomena for loss in training 
- **Gradual Decrease**: the most expected behavior is a gradual decrease in the loss over time as the the network learns to fit the training data better.
- **Fluctuations**: small fluctuations in the loss are normal and can be attributed to: 
    - Stochastic nature of mini-batch training 
    - Gradient noise, especially with `small batch size`
    - Learning rate effects, particularly if it's too high 
- **Loss spike**: sudden increases in loss, known as spikes, can occur due to: 
    - Lower-loss-as-sharper (LLAS) structure in the loss landscape
    - instability when entering sharp regions of the landscape
    - high learning rates cause drastic parameter updates

    - `Mitigation strategies`: 
        - learning rate adjustment 
        - increase batch size 
        - batch normalization 
        - gradient clipping 
- **Overfitting Signs**: 
    - Training loss continues to decrease while validation loss increases or plateaus 
    - Increasing gap between training and validation loss over time 

    - `Mitigation strategies`: 
        - a
- **Underfitting Indicators**: 
    - Both training and validation loss remain high 
    - Validation loss is significantly higher than training loss from start
- **Periodic behavior**: in some cases, the loss may show periodic increases or decreases, which could be due to: 
    - Learning rate schedules like cyclical learning rates
    - Batch normalization effects
    - Dataset ordering if not properly shuffled 
- **Vanishing Gradient**: in deep neural networks, especially those with sigmoid or tahn activations 
    - Loss may stagnate early in training 
    - Earlier layers may show little to no change in weights 
- **Exploding Gradient**: 
    - Sudden spikes in loss values 
    - Unstable or diverging training process
- **Loss of Plasticity**: Over extended training, especially in continual learning scenarios: 
    - Gradual decrease in the network's ability to learn new information 
    - Performance on new tasks may degrade to that of a linear network 

