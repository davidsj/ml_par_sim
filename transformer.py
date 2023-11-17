from lib import *
from math import floor, ceil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=2048)
parser.add_argument('--num_experts', type=int, default=1)
parser.add_argument('--seq_len', type=int, default=2048)
parser.add_argument('--use_glu', action='store_true', default=False)
parser.add_argument('--no_multi_query_attn', action='store_true', default=False)
parser.add_argument('--activation_checkpointing', action='store_true', default=False)
parser.add_argument('--consecutive_attn_and_ffn', action='store_true', default=False)
parser.add_argument('--dataset_size', type=float, required=True)
parser.add_argument('--num_layers', type=int, required=True)
parser.add_argument('--d_model', type=int, required=True)
parser.add_argument('--d_ffn_ratio', type=int, default=None)
parser.add_argument('--d_ffn', type=int, default=None)
parser.add_argument('--d_head', type=int, default=64)
parser.add_argument('--num_heads', type=int, default=None)
parser.add_argument('--total_attn_width_ratio', type=float, default=None)
parser.add_argument('--p_data', type=int, default=1)
parser.add_argument('--p_pipe', type=int, default=1)
parser.add_argument('--p_exp', type=int, default=1)
parser.add_argument('--p_seq', type=int, default=1)
parser.add_argument('--p_tensor_int', type=int, default=1)
parser.add_argument('--p_tensor_ext', type=int, default=1)
parser.add_argument('--cluster_flop_rate', type=float, default=None)
parser.add_argument('--skip_efficiency_check', action='store_true', default=False)
parser.add_argument('--fwd_only', action='store_true', default=False)
parser.add_argument('--net_precision_bytes', type=int, default=2)
args = parser.parse_args()

e = args.num_experts
b = args.batch_size
l = args.seq_len
expert_l = div(l, e)
multiq = not args.no_multi_query_attn
L = args.num_layers
d = args.d_model

assert not (args.d_ffn_ratio is not None and args.d_ffn is not None), \
    'Cannot specify both d_ffn_ratio and d_ffn'
if args.d_ffn is None:
    if args.d_ffn_ratio is None:
        d_ffn = 4*d
    else:
        d_ffn = args.d_ffn_ratio*d
else:
    d_ffn = args.d_ffn


d_head = args.d_head
assert not (args.num_heads is not None and args.total_attn_width_ratio is not None), \
    'Cannot specify both num_heads and total_attn_width_ratio'
if args.num_heads is None:
    if args.total_attn_width_ratio is None:
        num_heads = div(d, d_head)
    else:
        num_heads = div(d*args.total_attn_width_ratio, d_head)
else:
    num_heads = args.num_heads

p_data = args.p_data
p_pipe = args.p_pipe
p_exp = args.p_exp
p_seq = args.p_seq
p_tint = args.p_tensor_int
p_text = args.p_tensor_ext

max_n_microbatches = div(b, p_data)

# TODO: Take into account interleaved pipeline stage schedule as per https://arxiv.org/abs/2104.04473
if not args.skip_efficiency_check:
    # Keep batch size per expert reasonable (below a critical noise scale).
    # TODO: Might be dubious to assume you can scale this way.
    assert b*expert_l <= 4096*2048,\
        f'Tokens per batch too large: b*expert_l = {b}*{expert_l} = {b*expert_l} > {2048*2048} = 2048*2048'

    if p_pipe > 1:
        # Keep pipeline bubble small.
        min_efficient_num_microbatches = 4*p_pipe
        max_efficient_microbatch_size = floor(b/min_efficient_num_microbatches)
        assert max_n_microbatches >= min_efficient_num_microbatches, \
            f'Pipeline bubble too large: {max_n_microbatches} < {min_efficient_num_microbatches}'
    else:
        max_efficient_microbatch_size = b

    # Maintain arithmetic intensity (1024 tokens per matmul).
    tokens_per_matmul_per_sample = expert_l/(p_data*p_seq)
    min_efficient_microbatch_size = 1024/tokens_per_matmul_per_sample
    min_efficient_microbatch_size = ceil(min_efficient_microbatch_size/p_data)*p_data # has to be multiple of p_data
    assert min_efficient_microbatch_size <= max_efficient_microbatch_size, \
        f'Efficient microbatch size impossible: {min_efficient_microbatch_size} > {max_efficient_microbatch_size}'
    allowed_microbatch_sizes = [mbsz for mbsz in range(min_efficient_microbatch_size, \
                                                      max_efficient_microbatch_size+1, p_data) if b % mbsz == 0]
    assert len(allowed_microbatch_sizes) > 0, \
        f'No factor of {b} and multiple of {p_data} found between {min_efficient_microbatch_size} and {max_efficient_microbatch_size}'

    # Maintain arithmetic intensity (1024 residual and hidden units per FFN matmul).
    assert p_tint <= d_ffn/1024, \
        f'Internal tensor parallelism too large: {p_tint} > d_ffn/1024 = {d_ffn//1024}'
    assert p_tint <= num_heads*d_head/1024, \
        f'Internal tensor parallelism too large: {p_tint} > num_heads*d_head/1024 = {num_heads*d_head//1024}'
    assert p_text <= d/1024, \
        f'External tensor parallelism too large: {p_text} > {d//1024}'

n_batches = round(args.dataset_size / (b*l))
D = n_batches*b*l # number of tokens

p = p_pipe*p_data*p_exp*p_seq*p_tint*p_text

# Parallel Dimensions
DATA = ParallelAxis('DATA', p_data)
PIPE = ParallelAxis('PIPE', p_pipe, rotational=True)
EXP = ParallelAxis('EXP', p_exp)
SEQ = ParallelAxis('SEQ', p_seq)
TENSOR_EXT = ParallelAxis('TENSOR_EXT', p_text)
TENSOR_INT = ParallelAxis('TENSOR_INT', p_tint)

# Data Dimensions
assert b % DATA.deg == 0
SAMPLE = DataDim('SAMPLE', DATA.deg, DATA)
LAYER = DataDim('LAYER', PIPE.deg, PIPE, virtual_size=L)
EXPERT = DataDim('EXPERT', e, EXP)
TOKEN = DataDim('TOKEN', expert_l, SEQ)
WRITE_EXPERT = DataDim('WRITE_EXPERT', e, autoreduce=True)
WRITE_TOKEN = DataDim('WRITE_TOKEN', TOKEN.size, TENSOR_EXT, autoreduce=True)

D_MODEL = DataDim('D_MODEL', d, TENSOR_EXT, autoreduce=True)
D_FFN = DataDim('D_FFN', d_ffn, TENSOR_INT, autoreduce=True)

ATTN_HEAD = DataDim('ATTN_HEAD', num_heads, TENSOR_INT)
D_HEAD = DataDim('D_HEAD', d_head, autoreduce=True)

# Full training run.
with Meter(n_batches):
    # Broadcasting of weights and reduction of their gradients.
    with Meter(div(L, LAYER.size)):
        # Attention weights
        Wq_s, Wo_s = [Data(LAYER, EXPERT, D_MODEL, ATTN_HEAD, D_HEAD, scatter=(DATA,SEQ)) for _ in range(2)]
        if multiq:
            Wk_s, Wv_s = [Data(LAYER, EXPERT, D_MODEL, D_HEAD, scatter=(DATA,SEQ,TENSOR_INT)) for _ in range(2)]
        else:
            Wk_s, Wv_s = [Data(LAYER, EXPERT, D_MODEL, ATTN_HEAD, D_HEAD, scatter=(DATA,SEQ)) for _ in range(2)]

        # FFN weights
        Wff1_s, Wff2_s = [Data(LAYER, EXPERT, D_MODEL, D_FFN, scatter=(DATA,SEQ)) for _ in range(2)]
        if args.use_glu:
            Wff1b_s = Data(LAYER, EXPERT, D_MODEL, D_FFN, scatter=(DATA,SEQ))

        size = 0
        for W in [Wq_s, Wo_s, Wk_s, Wv_s, Wff1_s, Wff2_s]:
            size += L/LAYER.size*prod(dim.size for dim in W.data_dims)
        if args.use_glu:
            size += L/LAYER.size*prod(dim.size for dim in Wff1b_s.data_dims)
        print(f'N: {size:.3e} ({size/e:.3e} effective)')
        print(f'D: {D:.3e}')
        if args.skip_efficiency_check:
            print(f'b: {b}')
        else:
            print(f'b: {b} (microbatch size must be in {allowed_microbatch_sizes} to be efficient)')
        print(f'P: {prod(par.deg for par in ParallelAxis.ALL)} = {ParallelAxis.ALL}')
        print()

        Wq, Wo, Wk, Wv = [Wq_s.bcast(), Wo_s.bcast(), Wk_s.bcast(), Wv_s.bcast()]
        Wff1, Wff2 = [Wff1_s.bcast(), Wff2_s.bcast()]
        if args.use_glu:
            Wff1b = Wff1b_s.bcast()

    # Forward and backward pass for all tokens in the batch.
    with Meter(b*l):
        # TODO: Not currently modeling embedding and unembedding.

        # Forward and backward pass for one token.
        with Meter(div(L, LAYER.size) / (p_data*l)):
            # Input
            x = Data(SAMPLE, LAYER, EXPERT, TOKEN, D_MODEL)

            # Attention
            q = Wq @ x

            # TODO: The write heads are broadcast against TENSOR_INT
            # (adding a small amount of unnecessary work in the
            # multi-query case), and receives are unmasked (adding
            # some unnecessary work in the masked decoder case).
            k = (Wk * x).reduce(D_MODEL, transpose=(TOKEN, WRITE_TOKEN)).scatter_transpose(EXPERT, WRITE_EXPERT).bcast()
            v = (Wv * x).reduce(D_MODEL, transpose=(TOKEN, WRITE_TOKEN)).scatter_transpose(EXPERT, WRITE_EXPERT).bcast()

            qk = q @ k
            softmax_normalizer = qk.reduce(WRITE_EXPERT).allreduce(WRITE_TOKEN)
            a = (qk * softmax_normalizer) @ v

            attn_out = Wo @ a # not yet reduced by ATTN_HEAD

            # By default, the FFN is computed logically in parallel
            # with self-attention, so it doesn't need its output. But
            # if args.consecutive_attn_and_ffn, we need an allreduce
            # before moving forward.
            if args.consecutive_attn_and_ffn:
                x = attn_out.allreduce(ATTN_HEAD)

            # Feed-Forward Network
            h = Wff1 @ x
            if args.use_glu:
                h2 = Wff1b @ x
                h = h * h2
            ffn_out = Wff2 * h # not yet reduced by D_FFN

            # Residual
            #
            # Note the actual addition of x is a low-dimension
            # pointwise operation, so we don't bother to model it.
            #
            # Currently not modeling the router network, but that
            # should be inexpensive.
            #
            # TODO: Support for routing to multiple experts (k > ).
            if args.consecutive_attn_and_ffn:
                x = ffn_out.allreduce(D_FFN, rotate=(LAYER, EXPERT))
            else:
                x = attn_out.reduce_with(ffn_out, dims=(ATTN_HEAD, D_FFN)).bcast(rotate=(LAYER, EXPERT))

print()
print('FULL TRAINING RUN COST')
Meter.print(args.cluster_flop_rate, p,
            activation_checkpointing=args.activation_checkpointing,
            fwd_only=args.fwd_only, net_precision=args.net_precision_bytes)

# counter = Meter.get()
# print(f"{p}, {counter[('FLOP', 'total')]/p}, {counter[('net', 'total')]}")
