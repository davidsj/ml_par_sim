# Model the Compute and Inter-Cluster Bandwidth for a Large ML Training Run

`lib.py` contains general utilities to specify a model architecture and parallelization scheme, and automatically determine the compute and bandwidth costs associated with the forward and backward passes.

By default, constraints on arithmetic intensity and gradient noise scale are enforced to ensure good utilization of tensor processors and efficient use of compute.

`transformer.py` applies these utilities to model a training run for a GPT-like language model.

An example of simulating a GPT-4-like training run on A100's at 50% utilization (in this case, the "clusters" are individual A100's):

    $ python3 http://transformer.py --cluster_flop_rate 156e12 --num_experts 8 --batch_size 8192 --seq_len 8192 --dataset_size 13e12 --num_layers 120 --d_model 15360 --p_pipe 120 --p_data 16 --p_seq 1 --p_exp 8 --p_tensor_int 1 --p_tensor_ext 1
    N: 2.267e+12 (2.834e+11 effective)
    D: 1.300e+13
    b: 8192 (microbatch size must be in [16] to be efficient)
    P: 15360 = {TENSOR_EXT: 1, DATA: 16, TENSOR_INT: 1, SEQ: 1, EXP: 8, PIPE: 120}
    
    FULL TRAINING RUN COST
        Metric            Fwd            Bwd          Total
          FLOP      8.155e+24      1.632e+25      2.447e+25
            Tx      2.458e+19      3.192e+19      5.650e+19
            Rx      3.192e+19      3.192e+19      6.384e+19
           net      5.650e+19      6.384e+19      1.203e+20
    
    Cluster FLOP/sec:    1.560e+14
    Cluster bytes/sec:   1.534e+09 (7.203e+08 Tx, 8.139e+08 Rx)
    
    Total FLOP/sec:      2.396e+18
    Total bytes/sec:     2.357e+13 (1.106e+13 Tx, 1.250e+13 Rx)
    
    Runtime:             118.2 days
