## Problem (benchmarking_script): 4 points

(b) Time the forward and backward passes for the model sizes described in §1.1.2. Use 5 warmup steps and compute the average and standard deviation of timings over 10 measurement steps. How long does a forward pass take? How about a backward pass? Do you see high variability across measurements, or is the standard deviation small?

- With my hyperparameters and on mps forward takes 0.092 seconds and backward takes 0.196 seconds. Standard deviation is quite small.

(c) One caveat of benchmarking is not performing the warm-up steps. Repeat your analysis without the warm-up steps. How does this affect your results? Why do you think this happens? Also try to run the script with 1 or 2 warm-up steps. Why might the result still be different?

- Having no warm-up steps increases the time and standard deviation. First iteration are used to prepare GPU for training. 1 and 2 warm-up steps bring the standard deviation back to the old one on mps. 

## Problem (nsys_profile): 5 points
(a) What is the total time spent on your forward pass? Does it match what we had measured before with the Python standard library?

Deliverable: Forward pass with large model takes around 80 ms on RTX 5070 Ti

## Problem (benchmarking_mixed_precision): 2 points
(a) Consider the following model:
```python
class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__() 
        self.fc1 = nn.Linear(in_features, 10, bias=False) 
        self.ln = nn.LayerNorm(10) 
        self.fc2 = nn.Linear(10, out_features, bias=False) 
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x)) 
        x = self.ln(x) 
        x = self.fc2(x) 
        return x
```

Suppose we are training the model on a GPU and that the model parameters are originally in FP32. We’d like to use autocasting mixed precision with FP16. What are the data types of:

- the model parameters within the autocast context,

- the output of the first feed-forward layer (ToyModel.fc1),

- the output of layer norm (ToyModel.ln),

- the model’s predicted logits,

- the loss,

- and the model’s gradients?

Deliverable:

- FP32
- FP16
- FP32
- FP16
- FP32
- FP32

(b) You should have seen that FP16 mixed precision autocasting treats the layer normalization layer differently than the feed-forward layers. What parts of layer normalization are sensitive to mixed precision? If we use BF16 instead of FP16, do we still need to treat layer normalization differently? Why or why not?

Deliverable: In layer norm we need to compute $\mu$ and $\sigma^2$, which require accumulation operations that can suffer from precision loss in FP16. BF16 would help us avoid some of the overflows/underflows.


## Problem (memory_profiling): 4 points

(b) What is the peak memory usage of each context length when doing a forward pass? What about when doing a full training step?

| context length | forward | forward + backward |
| --- | --- | --- |
| 512 | 6 GB | 13 GB |
| 256 | 6 GB | 11 GB |
| 128 | 6 GB | 10.3 GB|

(c) Find the peak memory usage of the large model when using mixed-precision, for both a forward pass and a full optimizer step. Does mixed-precision significantly affect memory usage?

Deliverable: Without mixed: 12.3 GB, with mixed precision: 12.9 GB. So mixed precision actually increases peak memory.

## Problem (naive_ddp_benchmarking): 3 points
Step 1 took 0.21 seconds
Communication took 0.13 seconds
Communication overhead: 60.63%
Step 2 took 0.21 seconds
Communication took 0.13 seconds
Communication overhead: 60.46%
Step 3 took 0.21 seconds
Communication took 0.13 seconds
Communication overhead: 60.51%
Step 4 took 0.21 seconds
Communication took 0.13 seconds
Communication overhead: 60.54%