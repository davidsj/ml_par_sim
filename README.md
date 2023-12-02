# Model the Compute and Inter-Cluster Bandwidth for a Large ML Training Run

`lib.py` contains general utilities to specify a model architecture and parallelization scheme, and automatically determine the compute and bandwidth costs associated with the forward and backward passes.

By default, constraints on arithmetic intensity and gradient noise scale are enforced to ensure good utilization of tensor processors and efficient use of compute.

`transformer.py` applies these utilities to model a training run for a GPT-like language model.

An example of simulating a GPT-4-like training run on A100's at 50% utilization (in this case, the "clusters" are individual A100's):

    $ python3 transformer.py --cluster_flop_rate 156e12 --num_experts 8 --batch_size 8192 --seq_len 8192 --dataset_size 13e12 --num_layers 120 --d_model 15360 --p_pipe 15 --p_data 16 --p_seq 1 --p_exp 8 --p_tensor_int 8 --p_tensor_ext 1
    N: 2.267e+12 (2.834e+11 effective)
    D: 1.300e+13
    b: 8192 (microbatch size must be in [16, 32, 64, 128] to be efficient)
    P: 15360 = {TENSOR_EXT: 1, DATA: 16, TENSOR_INT: 8, SEQ: 1, EXP: 8, PIPE: 15}
    
    
    FULL TRAINING RUN COST
        Metric            Fwd            Bwd          Total
          FLOP      8.198e+24      1.640e+25      2.460e+25
            Tx      1.923e+20      2.308e+20      4.231e+20
            Rx      3.746e+20      3.746e+20      7.492e+20
           net      5.669e+20      6.054e+20      1.172e+21
    
    Cluster FLOP/sec:    1.560e+14
    Cluster bytes/sec:   1.487e+10 (5.367e+09 Tx, 9.501e+09 Rx)
    
    Total FLOP/sec:      2.396e+18
    Total bytes/sec:     2.284e+14 (8.243e+13 Tx, 1.459e+14 Rx)
    
    Runtime:             118.8 days
