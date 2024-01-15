# A simple parallel MoE transformer training simulation, considering
# only the weight matrix computations and data-, pipeline-, tensor-,
# and expert-parallel communication.

# For now we assume no activation checkpointing, network fabric
# support for all-reduce, perfectly balanced expert routing, and an
# expert-parallel degree equal to the number of experts.

import argparse

def matmul(a, b, c, chip, base_util, precision):
    """Calculate number of FLOP and duration of a matmul of shape (a x b) x (b x c)."""
    flop = 2*a*b*c
    mem_io = a*b + b*c + a*c
    
    flop_time = flop / (base_util * chip['flop_per_sec_8bit']) * (precision/8)
    mem_io_time = mem_io / chip['io']
    time = max(flop_time, mem_io_time)
    return flop, time

def allreduce(size, scatter_degree, allreduce_degree, chip, precision):
    """Calculate duration of all-reduce."""
    if allreduce_degree == 1:
        return 0
    else:
        return tx_rx(size, scatter_degree, chip, precision) # (network fabric does actual reduction)

def tx_rx(size, scatter_degree, chip, precision):
    """Calculate duration of a point-to-point send and receive."""
    bytes_per_chip = size/scatter_degree * precision/8
    time = 2 * bytes_per_chip / chip['net'] # tx and rx
    return time

chip_types = dict(H100=dict(flop_per_sec_8bit=1979e12, io=3.35e12, net=900e9),
                  A100_40GB=dict(flop_per_sec_8bit=624e12, io=1555e9, net=600e9),
                  H100_x64_pod=dict(flop_per_sec_8bit=64*1979e12, io=64*3.35e12, net=1e9),
                  RTX4090=dict(flop_per_sec_8bit=165.16e12, io=1008e9, net=10e9)) # unclear bandwidth

args = argparse.ArgumentParser()
# Model Shape
# include helpful docs
args.add_argument('--precision_bits', type=int, default=8, dest='prec', help='precision in bits')
args.add_argument('--n_layers', type=int, required=True, dest='L', help='number of layers')
args.add_argument('--d_model', type=int, required=True, dest='d', help='residual width')
args.add_argument('--d_internal_ratio', type=int, default=5, dest='d_internal_ratio',
                  help='internal width as a multiple of residual width; '
                       'equal to d_ffn + the combined attention head width')
args.add_argument('--n_experts', type=int, default=1, dest='e',
                  help='number of experts, and also the expert-parallel degree')
args.add_argument('--n_routed_experts', type=int, default=1, dest='k',
                  help='number of experts to route to per token')

# Dataset Size
args.add_argument('--dataset_size', type=float, default=None, dest='D',
                    help='number of tokens in the dataset; '
                         'if not specified, defaults to 20x the number of parameters')

# Data Shape
args.add_argument('--batch_size', type=int, default=2048, dest='b',
                    help='number of sequences per batch')
args.add_argument('--microbatch_size', type=int, default=1, dest='m',
                    help='number of sequences per microbatch, i.e. '
                         'handled by one pipeline stage at a time '
                         '(across all replicas)')
args.add_argument('--seq_len', type=int, default=2048, dest='l',
                    help='tokens per sequence')

# Hardware
args.add_argument('--chip_type', type=str, default='H100', choices=chip_types.keys())
args.add_argument('--network_bandwidth', type=float, default=None,
                    help='network bandwidth (B/s); '
                         'if not specified, defaults to the chip\'s network bandwidth',
                    dest='net')
args.add_argument('--base_utilization', type=float, default=0.6,
                    help='utilization that would be achieved by a single '
                         'matmul on a single chip')

# Parallelism
args.add_argument('--pp', type=int, default=1,
                    help='number of pipeline stages')
args.add_argument('--pipeline_interleaving', type=int, default=1,
                    help='number of times a given pipeline stage sees the same sequence')
args.add_argument('--dp', type=int, default=1,
                    help='data-parallel degree')
args.add_argument('--tp', type=int, default=1,
                    help='internal (d_int) tensor-parallel degree')
args.add_argument('--tp2', type=int, default=1,
                    help='external (d_model) tensor-parallel degree')

# Parse args.
args = args.parse_args()
b, m, l, prec, L, d, d_internal_ratio, e, k, D, chip_type, pp, pipeline_interleaving, dp, tp, tp2, base_util, net = \
    args.b, args.m, args.l, args.prec, args.L, args.d, args.d_internal_ratio, args.e, args.k, args.D, args.chip_type, args.pp, args.pipeline_interleaving, \
    args.dp, args.tp, args.tp2, args.base_utilization, args.net
d_int = d_internal_ratio*d
ep = e # for now we assume ep == e
p = pp * dp * tp * tp2 * ep

params_per_expert = L*2*d*d_int
N = k*params_per_expert
P = e*params_per_expert
if D is None:
    D = 20*P
n_batches = D // (b*l)
n_microbatches_per_batch = b // m
D = n_batches * b*l
chip = chip_types[args.chip_type]
cluster_flop_per_sec = p * chip['flop_per_sec_8bit'] * base_util * (8/prec)
if net is not None:
    chip['net'] = net

# Validate args.
assert e % k == 0, 'routed experts must divide number of experts'
assert b % m == 0, 'microbatch size must divide batch size'
assert (m*l*k) % (dp*e) == 0, 'need integer number of tokens per matmul'
assert L % (pp * pipeline_interleaving) == 0, 'need integer number of pipeline passes'
assert (pp > 1) or (pipeline_interleaving == 1), 'can\'t interleave a single pipeline stage'
assert (m*l) % dp == 0, 'data-parallel degree must divide microbatch size (in tokens)'
assert d_int % tp == 0, 'tensor-parallel degree must divide internal dimension'
assert d % tp2 == 0, 'exterior tensor-parallel degree must divide residual dimension'
assert e % ep == 0, 'expert-parallel degree must divide number of experts'

# Print run info.
print(f'Model Shape: {L} x ({d} x {d_int}) (top {k} of {e} routing)')
print(f'Parameters: {P:.2e} ({N:.2e} per token), Training Tokens: {D:.2e}')
print(f'{p} {chip_type}s: ({chip["flop_per_sec_8bit"]/1e12:.1f} 8-bit TFLOP/s; {chip["io"]/1e12:.2f} TB/s IO; {round(chip["net"]/1e8)/10} GB/s network), Base Utilization: {base_util:.2f}')

# Calculate pipeline shape.
n_pipeline_steps = pp * pipeline_interleaving
pipe_boundary_intvl = L // n_pipeline_steps
pipeline_bubble_layers = (pp - 1) * pipe_boundary_intvl
print(f'Pipeline steps: {pp} stages x {pipeline_interleaving} interleaving; boundary interval: {pipe_boundary_intvl}; bubble layers: {pipeline_bubble_layers}')

# Calculate batch stats.
flop_per_batch = 6*N*b*l
print(f'FLOP per batch: {flop_per_batch:.2e}')
tokens_per_matmul = (m*l*k)//(dp*e)
experts_per_chip = e//ep
matmuls_per_chip_per_layer = 2*e//ep
d_int_per_matmul = d_int//tp
d_per_matmul = d//tp2
print(f'Matmuls per chip per layer: 2 matrices x {e//ep} experts = {matmuls_per_chip_per_layer}')
print(f'Matmul shape: {d_int_per_matmul} d_int x {d_per_matmul} d_model x {tokens_per_matmul} tokens')

_, matmul_time = matmul(d_int_per_matmul, d_per_matmul, tokens_per_matmul, chip, base_util, prec)
pipeline_bubble_time = 0

# Determine pipeline bubble time.
for layer in range(pipeline_bubble_layers):
    # First matmul.
    pipeline_bubble_time += 3*experts_per_chip*matmul_time # 1 fwd, 2 back
    # Internal reduce.
    pipeline_bubble_time += 2*allreduce(m*l*d_int * k, dp*tp*ep, tp2, chip, prec) # 1 fwd, 1 back
    # Second matmul.
    pipeline_bubble_time += 3*experts_per_chip*matmul_time # 1 fwd, 2 back
    # External reduce.
    # For now this is simplified by the assumption ep == e.
    pipeline_bubble_time += 2*allreduce(m*l*d, dp*tp2*ep//k, k*tp, chip, prec) # 1 fwd, 1 back

    # Crossing a pipeline stage or expert boundary. In this case the
    # previous external all-reduce can be assumed to be a
    # reduce-scatter before crossing the boundary, then a bcast after
    # crossing the boundary.
    if (layer % pipe_boundary_intvl) == (pipe_boundary_intvl - 1):
        pipeline_bubble_time += 2*tx_rx(m*l*d, dp*tp*tp2*ep, chip, prec)/2 # 1 fwd, 1 back, but we only care about the tx side
    elif (k*tp == 1) and (ep > 1):
        # There's some chance of expert-parallel communication even
        # without crossing a pipeline stage here. (If k*tp > 1 this
        # was already implicit in the all-reduce, since the network
        # fabric would have delivered it to the appropriate
        # expert-parallel ranks.)

        # TODO: A version of this which assumes
        # pp*pipeline_interleaving = L would simplify things, since
        # we'd always be crossing a pipeline stage after each layer.
        ep_prob = (ep - 1) / ep
        pipeline_bubble_time += 2*ep_prob*tx_rx(m*l*d, dp*tp*tp2*ep, chip, prec) # 1 fwd, 1 back

# Determine pipeline full time. Communication is overlapped with
# computaton as much as possible.
pipeline_full_steps = (L//pp) * n_microbatches_per_batch
pipeline_full_flop_step_time = 3 * matmuls_per_chip_per_layer * matmul_time # bwd pass is twice as costly
pipeline_full_flop_time = pipeline_full_steps * pipeline_full_flop_step_time
pipeline_full_net_step_time = 0 # TODO: Figure this out.

# Internal reduce.
pipeline_full_net_step_time += 2*allreduce(m*l*d_int * k, dp*tp*ep, tp2, chip, prec) # 1 fwd, 1 back
# External reduce. For now this is simplified by the assumption ep == e.
pipeline_full_net_step_time += 2*allreduce(m*l*d, dp*tp2*ep//k, k*tp, chip, prec) # 1 fwd, 1 back
# Not currently modeling expert or pipeline stage boundary crossing.

pipeline_full_net_time = pipeline_full_steps * pipeline_full_net_step_time
pipeline_full_time = max(pipeline_full_flop_time, pipeline_full_net_time)

# Determine time spent doing data-parallel communication.
if dp == 1:
    gradient_reduction_time = 0
else:
    gradient_reduction_time = allreduce(P, p//dp, dp, chip, prec)

batch_time = pipeline_bubble_time + pipeline_full_time + gradient_reduction_time
print()
print('BATCH TIME')
print(f'Pipeline bubble time: {pipeline_bubble_time:9.4f} secs')
print(f'Pipeline full time:   {pipeline_full_time:9.4f} secs (FLOP: {pipeline_full_flop_time:9.4f} secs, net: {pipeline_full_net_time:9.4f} secs)')
print(f'Gradient reduction:   {gradient_reduction_time:9.4f} secs')
print(f'Total batch time:     {batch_time:9.4f} secs')

util = base_util * flop_per_batch / (cluster_flop_per_sec * batch_time)
print()
print(f'Utilization:          {util:9.4f}')


total_flop = 6*N*D
flop_per_expert = total_flop/e
print()
print('TOTAL')
print(f' Run time: {batch_time * n_batches / 86400:6.1f} days')
print(f' FLOP:             {total_flop:.2e}')
print(f' FLOP per expert:  {flop_per_expert:.2e}')
