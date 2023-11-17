# TODO: Unit tests.

from typing import Iterable, List, Optional, Set, Tuple
from math import prod
from collections import Counter

class Meter():
    counter_stack: List[Tuple[Counter, float]] = [(Counter(), 1)]

    def __init__(self, multiplier: float = 1):
        self.counter: Counter = Counter()
        self.multiplier = multiplier

    def __enter__(self):
        Meter.counter_stack.append((self.counter, self.multiplier))
        return None

    def __exit__(self, *args):
        child, mult = Meter.counter_stack.pop()
        parent, _ = Meter.counter_stack[-1]
        for key, value in child.items():
            parent[key] += mult*value

    @staticmethod
    def get():
        return Meter.counter_stack[-1][0]

    @staticmethod
    def print(cluster_flop_per_sec: Optional[float] = None,
              p: Optional[int] = None,
              net_precision: int = 2,
              activation_checkpointing: bool = False,
              fwd_only: bool = False):
        # BUG: This method mutates the underlying counter, so it's not
        # safe to call more than once or for any but the top-level
        # counter.

        counter = Meter.get()
        if fwd_only:
            print(f'{"Metric":>10}    {"Fwd":>11}')
        else:
            print(f'{"Metric":>10}    {"Fwd":>11}    {"Bwd":>11}    {"Total":>11}')
        for direc in ['fwd', 'bwd']:
            counter[('net', direc)] = counter[('Tx', direc)] + counter[('Rx', direc)]
        for metric in ['FLOP', 'Tx', 'Rx', 'net']:
            if activation_checkpointing:
                counter[(metric, 'bwd')] += counter[(metric, 'fwd')]
            line = f'{metric:>10}'
            counter[(metric, 'total')] = counter[(metric, 'fwd')] + counter[(metric, 'bwd')]
            direcs = ['fwd'] if fwd_only else ['fwd', 'bwd', 'total']
            for direc in direcs:
                cnt = counter[(metric, direc)]
                line += f'    {cnt:>11.3e}'
                # for suffix, power in reversed(list(zip([' ', 'k', 'M', 'G', 'T', 'P', 'E'], range(0, 21, 3)))):
                #     if power == 0 or cnt >= 10**power:
                #         line += f'    {cnt/10**power:>10.3f}{suffix}'
                #         break
            print(line)

        total_direc = 'fwd' if fwd_only else 'total'
        net_per_flop = counter[('net', total_direc)] / counter[('FLOP', total_direc)]
        tx_bytes_per_flop = net_precision * counter[('Tx', total_direc)] / counter[('FLOP', total_direc)]
        rx_bytes_per_flop = net_precision * counter[('Rx', total_direc)] / counter[('FLOP', total_direc)]
        net_bytes_per_flop = tx_bytes_per_flop + rx_bytes_per_flop
        if cluster_flop_per_sec is not None:
            print()
            print(f'Cluster FLOP/sec:    {cluster_flop_per_sec:.3e}')
            print(f'Cluster bytes/sec:   {net_bytes_per_flop*cluster_flop_per_sec:.3e} ({tx_bytes_per_flop*cluster_flop_per_sec:.3e} Tx, {rx_bytes_per_flop*cluster_flop_per_sec:.3e} Rx)')
            if p is not None:
                flop_per_sec = p * cluster_flop_per_sec
                runtime_sec = counter[('FLOP', total_direc)] / flop_per_sec
                print()
                print(f'Total FLOP/sec:      {flop_per_sec:.3e}')
                print(f'Total bytes/sec:     {net_bytes_per_flop*flop_per_sec:.3e} ({tx_bytes_per_flop*flop_per_sec:.3e} Tx, {rx_bytes_per_flop*flop_per_sec:.3e} Rx)')
                print()
                if runtime_sec >= 86400:
                    print(f'Runtime:             {runtime_sec/86400:.1f} days')
                elif runtime_sec >= 3600:
                    print(f'Runtime:             {runtime_sec/3600:.1f} hours')
                elif runtime_sec >= 60:
                    print(f'Runtime:             {runtime_sec/60:.1f} minutes')
                else:
                    print(f'Runtime:             {runtime_sec:.1f} seconds')

def ensure_set(a: Iterable) -> Set:
    cnt = Counter(a)
    for k, v in cnt.items():
        assert v == 1, (k, v)
    return set(cnt)

def div(a: int, b: int) -> int:
    """Divide a by b, ensuring that the result is an integer."""
    assert a % b == 0, (a, b)
    return a // b

# TODO: Make parallel axes and data dimensions symbolic rather than
# numeric, so we can directly yield formulas in terms of them.
class ParallelAxis():
    # TODO: Comment this class.
    next_id = 0
    ALL: Set['ParallelAxis'] = set()
    def __init__(self, name: str, deg: int, rotational:bool = False):
        self.name = name
        self.deg = deg
        self.rotational = rotational
        self.id = ParallelAxis.next_id
        ParallelAxis.next_id += 1
        ParallelAxis.ALL.add(self)

    def __repr__(self) -> str:
        return f'{self.name}: {self.deg}'

class DataDim():
    # TODO: Comment this class.
    next_id = 0
    def __init__(self, name: str, size: int,
                 pax: Optional[ParallelAxis] = None,
                 virtual_size: Optional[int] = None,
                 autoreduce: bool = False):
        self.name = name
        self.size = size
        self.pax = pax
        self.autoreduce = autoreduce
        if pax is None:
            assert virtual_size is None, virtual_size
            self.local_size = size
        elif pax.rotational:
            assert size == pax.deg, (size, pax.deg)
            assert virtual_size is not None, virtual_size
            assert virtual_size % pax.deg == 0, (virtual_size, pax.deg)
            self.local_size = 1
            self.virtual_size = virtual_size
        else:
            assert virtual_size is None, virtual_size
            self.local_size = div(size, pax.deg)

        self.id = DataDim.next_id
        DataDim.next_id += 1

    def __repr__(self):
        ret = f'{self.name}: {self.size}'
        if self.pax is not None and self.pax.rotational:
            ret += f'/{self.virtual_size}'
        return ret

    def __add__(self, other: 'DataDim') -> 'DataDim':
        assert self.pax == other.pax, (self, other)
        assert self.pax is None or not self.pax.rotational, self
        return DataDim(f'{self.name}+{other.name}', self.size + other.size, self.pax)

class Data():
    # TODO: Comment this class.
    def __init__(self, *dims: DataDim, scatter: Iterable[ParallelAxis] = ()):
        self.data_dims = ensure_set(dims)
        self.paxs = ensure_set(dim.pax for dim in self.data_dims
                                   if dim.pax is not None)
        self.scatter_axs = ensure_set(scatter)

        # Validate that the parallel axes are disjoint and contained
        # in ParallelAxis.ALL.
        assert self.paxs.isdisjoint(self.scatter_axs), (self.paxs, self.scatter_axs)
        assert self.paxs <= ParallelAxis.ALL, (self.paxs, ParallelAxis.ALL)
        assert self.scatter_axs <= ParallelAxis.ALL, (self.scatter_axs, ParallelAxis.ALL)

        # Validate that the scattering degree divides the size to be
        # scattered.
        local_size = prod(dim.local_size for dim in self.data_dims)
        scatter_deg = prod(parallel.deg for parallel in self.scatter_axs)
        assert local_size % scatter_deg == 0, (local_size, scatter_deg)

        # Broadcast on the remaining parallel axes.
        self.bcast_axs = ParallelAxis.ALL - (self.paxs | self.scatter_axs)

    def __repr__(self):
        ret = '('
        ret += ', '.join(map(repr, sorted(self.data_dims, key=lambda dim: dim.id)))
        if len(self.bcast_axs) > 0:
            ret += '; bcast='
            ret += ', '.join(map(repr, sorted(self.bcast_axs, key=lambda par: par.id)))
        if len(self.scatter_axs) > 0:
            ret += '; scatt='
            ret += ', '.join(map(repr, sorted(self.scatter_axs, key=lambda par: par.id)))
        ret += ')'
        return ret

    def cat(self, other: 'Data', cat_dims: Tuple[DataDim, DataDim]) -> Tuple['Data', DataDim]:
        # For now, no scattering is allowed (otherwise there's a
        # question of whether the two Datas are scattered the same
        # way).
        assert len(self.scatter_axs) == 0, self
        assert len(other.scatter_axs) == 0, other

        # Verify the concatenated dimensions are present and that they
        # constitute the only differences.
        my_dim, other_dim = cat_dims
        cat_dim = my_dim + other_dim
        assert self.data_dims - other.data_dims == {my_dim}, (self, other)
        assert other.data_dims - self.data_dims == {other_dim}, (self, other)

        new_dims = (self.data_dims - {my_dim}) | {cat_dim}
        return Data(*new_dims), cat_dim

    def reduce_with(self, other: 'Data', dims: Tuple[DataDim, DataDim]) -> 'Data':
        catted, cat_dim = self.cat(other, dims)
        return catted.reduce(cat_dim)

    def __matmul__(self, other: 'Data') -> 'Data':
        return self.dot(other)

    def dot(self, other: 'Data', transpose: Optional[Tuple[DataDim, DataDim]] = None) -> 'Data':
        """Pointwise multiplication and allreduce on autoreduced dimensions,
        i.e.  basically a matrix multiplication.

        transpose: Provided during allreduce; see Data.reduce for
        details.

        """
        # TODO: Either here or in __mult__, might need to think about
        # memory bandwidth implications so that arithmetic intensity
        # of different data movement patterns can be accounted for.
        ret = self * other

        # Do autoreductions, local dimensions first.
        # TODO: This could be even smarter, and deterministic.
        for dim in self.data_dims & other.data_dims:
            if dim.autoreduce and dim.pax is None:
                ret = ret.allreduce(dim, transpose=transpose)
        for dim in self.data_dims & other.data_dims:
            if dim.autoreduce and dim.pax is not None:
                ret = ret.allreduce(dim, transpose=transpose)
        return ret

    def __mul__(self, other: 'Data') -> 'Data':
        """Pointwise multiplication with other."""
        dims = self.data_dims | other.data_dims

        # For now, no scattering is allowed (otherwise there's a
        # question of whether the two Datas are scattered the same
        # way).
        assert len(self.scatter_axs) == 0, self
        assert len(other.scatter_axs) == 0, other

        # Because scattering is disallowed, this implies all parallel
        # axes are saturated either by data dimensions or by
        # broadcasting.
        pars = ensure_set(dim.pax for dim in dims if dim.pax is not None)
        bcast = ParallelAxis.ALL - pars

        # Compute the FLOP in this pointwise multiplication. In the
        # forward direction, a local broadcast must occur on each
        # processor, but this is free, so the number of FLOP is just
        # the number of element indices for the actual multiplication.
        size = prod(dim.size for dim in dims) * prod(par.deg for par in bcast)
        Meter.get()[('FLOP', 'fwd')] += size

        # In the backward direction, the gradient needs to be computed
        # for both inputs, so it's double the cost. Plus the local
        # broadcasts have to be reversed and accumulated into the
        # running gradient sum (which might also have contributions
        # from other operations), which is a reduction which is almost
        # double the cost yet again.
        self_size = prod(dim.size for dim in self.data_dims) * prod(par.deg for par in self.bcast_axs)
        other_size = prod(dim.size for dim in other.data_dims) * prod(par.deg for par in other.bcast_axs)
        Meter.get()[('FLOP', 'bwd')] += 2*size # self gradient and reduction
        Meter.get()[('FLOP', 'bwd')] += 2*size # other gradient and reduction

        return Data(*dims)

    def bcast(self, rotate: Iterable[DataDim] = ()) -> 'Data':
        """Broadcast all scattered dimensions.

        rotate: If provided, the broadcast is rotated across the
          layers in a pipeline for the specified data dimensions,
          necessiting additional communication when crossing pipeline
          stages.
        """
        # Determine the frequency with which we cross parallel ranks.
        parallel_stay_freq = 1.0
        for dim in rotate:
            assert dim in self.data_dims, (dim, self.data_dims)
            if dim.pax is not None:
                if dim.pax.rotational:
                    # This is generally for layer-wise pipelining.
                    #
                    # There are pax.deg - 1 pipeline stage crossings
                    # across dim.virtual_size layers. We currently
                    # aren't accounting for data transfer into and out
                    # of the network, which is small because it's
                    # pre-embedding, just a few bytes per token
                    # typically.
                    parallel_stay_freq *= 1 - (dim.pax.deg-1)/dim.virtual_size
                else:
                    # This is generally for expert routing.
                    #
                    # Even the initial loading of the data needs to be
                    # routed, so there's no boundary effect like with
                    # pipelining.
                    parallel_stay_freq *= 1/dim.pax.deg
        parallel_boundary_freq = 1 - parallel_stay_freq

        # Compute the FLOP, Tx, and Rx for this broadcast.
        #
        # An element at index (i) broadcasts to indices (i, j, k),
        # where i is over data dimensions of combined size I, j is
        # over broadcast parallel axes of combined size J, and k is
        # over currently scattered (but soon to be broadcast) parallel
        # axes of combined size K.
        #
        # No computation is required, but there are I transmissions
        # (if K > 1 or we're rotating across a pipeline stage
        # boundary), and I*J*(K-1) receives, or I*J*K when crossing a
        # parallel rank.
        I = prod(dim.size for dim in self.data_dims)
        J = prod(par.deg for par in self.bcast_axs)
        K = prod(par.deg for par in self.scatter_axs)
        Meter.get()[('FLOP', 'fwd')] += 0
        Meter.get()[('Tx', 'fwd')] += I*max(K > 1, parallel_boundary_freq)
        Meter.get()[('Rx', 'fwd')] += I*J*(K + parallel_boundary_freq - 1)

        # The backward pass is the equivalent of a reduce, as the
        # gradient is the sum of the gradients of the broadcasted
        # values: ∂y/∂xᵢⱼ = ∑ₖ∂y/∂xᵢⱼₖ. We can assume that I*J of the
        # gradient has already been calculated at the source, so this
        # requires I*J*(K - 1) transmissions, receives, and additions,
        # or I*J*K when crossing a parallel rank.
        #
        # In the forward direction, we were able to "skip" the
        # pre-existing broadcast indices J and just transmit one copy
        # per unique data element to the new indices requiring the
        # data, but in the backward direction, no such optimization is
        # possible as all of the gradients are different. Furthermore,
        # in this step we can only reduce the gradient down to I*J
        # elements, rather than all the way down to I: we have to
        # preserve J, or else we'd be double-counting with the later
        # backward pass corresponding to the earlier initial
        # broadcasts of indices J.
        Meter.get()[('FLOP', 'bwd')] += I*J*(K + parallel_boundary_freq - 1)
        Meter.get()[('Tx', 'bwd')] += I*J*(K + parallel_boundary_freq - 1)
        Meter.get()[('Rx', 'bwd')] += I*J*(K + parallel_boundary_freq - 1)

        return Data(*self.data_dims)

    def reduce(self, dim: DataDim, transpose: Optional[Tuple[DataDim, DataDim]] = None) -> 'Data':
        assert dim in self.data_dims, (dim, self.data_dims)
        dims = self.data_dims - {dim}

        # Compute the FLOP, Tx, and Rx for this reduction.
        #
        # An element at index (i, j) reduces to index (i), where i is
        # over the non-reduced data dimensions and broadcast axes of
        # combined size I, and j is over the reduced data dimension of
        # size J. This requires I*(J - 1) additions total.
        #
        # On the other hand, it requires only I*(K - 1) transmissions
        # and receives, where K is the parallelism degree.
        I = prod(dim.size for dim in dims) * prod(par.deg for par in self.bcast_axs)
        J = dim.size
        K = dim.pax.deg if dim.pax is not None else 1
        Meter.get()[('FLOP', 'fwd')] += I*(J - 1)
        Meter.get()[('Tx', 'fwd')] += I*(K - 1)
        Meter.get()[('Rx', 'fwd')] += I*(K - 1)

        # The backward pass is the equivalent of a broadcast,
        # requiring no computation, as the gradient of each component
        # of a sum is the same as the sum itself: ∂y/∂∑ⱼxᵢⱼ = ∂y/xᵢⱼ
        #
        # Each of the I reduced values must be transmitted once (if K
        # > 1), and received K-1 times.
        Meter.get()[('FLOP', 'bwd')] += 0
        Meter.get()[('Tx', 'bwd')] += I*(K > 1)
        Meter.get()[('Rx', 'bwd')] += I*(K - 1)

        if transpose is not None:
            # In this case, a previously-existing dimension (from_dim)
            # commandeers the parallelism axis of the reduced data,
            # getting transposed into it and renamed (to to_dim).
            from_dim, to_dim = transpose
            assert from_dim.size == to_dim.size, (from_dim, to_dim)
            assert to_dim.pax == dim.pax, (to_dim, dim)

            # The reduce of dim is assumed to scatter w.r.t. from_dim
            # onto the target parallelism rank, so that no additional
            # data movement is required. At this point, from_dim is
            # laid out by *both* its old parallelism axis and its
            # new parallelism axis, so we can just relabel it.
            dims = (dims - {from_dim}) | {to_dim}

            # Theoretically, transpose could be its own operation, but
            # this would require keeping track of how data is
            # scattered across "unused" parallelism dimensions, which
            # we currently don't do.

        pars = ensure_set(dim.pax for dim in dims if dim.pax is not None)
        scatter_axs = ParallelAxis.ALL - (pars | self.bcast_axs)
        return Data(*dims, scatter=scatter_axs)

    def allreduce(self, dim: DataDim,
                  transpose: Optional[Tuple[DataDim, DataDim]] = None,
                  rotate: Iterable[DataDim] = ()) -> 'Data':
        return self.reduce(dim, transpose).bcast(rotate=rotate)

    def scatter_transpose(self, from_dim: DataDim, to_dim: DataDim) -> 'Data':
        """Rename a parallel data dimension and leave it scattered."""
        assert from_dim in self.data_dims, (from_dim, self.data_dims)
        assert to_dim not in self.data_dims, (to_dim, self.data_dims)
        assert from_dim.size == to_dim.size, (from_dim, to_dim)

        assert to_dim.pax is None, to_dim

        dims = (self.data_dims - {from_dim}) | {to_dim}
        scatter_axs = self.scatter_axs
        if from_dim.pax is not None:
            scatter_axs |= {from_dim.pax}
        return Data(*dims, scatter=scatter_axs)
