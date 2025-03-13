import h5py
import torch
import numpy as np
from time import time
from typing import List, Tuple, Union
from torch_geometric.nn.pool.consecutive import consecutive_cluster

from src.data.csr import CSRData, CSRBatch
from src.utils import has_duplicates, tensor_idx, load_tensor


__all__ = ['Cluster', 'ClusterBatch']


class Cluster(CSRData):
    """Child class of CSRData to simplify some common operations
    dedicated to cluster-point indexing.
    """

    __value_serialization_keys__ = ['points']
    __is_index_value_serialization_key__ = None

    def __init__(
            self,
            pointers: torch.Tensor,
            points: torch.Tensor,
            dense: bool = False,
            **kwargs):
        super().__init__(
            pointers, points, dense=dense, is_index_value=[True])

    @classmethod
    def get_base_class(cls) -> type:
        """Helps `self.from_list()` and `self.to_list()` identify which
        classes to use for batch collation and un-collation.
        """
        return Cluster

    @classmethod
    def get_batch_class(cls) -> type:
        """Helps `self.from_list()` and `self.to_list()` identify which
        classes to use for batch collation and un-collation.
        """
        return ClusterBatch

    @property
    def points(self) -> torch.Tensor:
        return self.values[0]

    @points.setter
    def points(self, points: torch.Tensor):
        assert points.device == self.device, \
            f"Points is on {points.device} while self is on {self.device}"
        self.values[0] = points
        # if src.is_debug_enabled():
        #     self.debug()

    @property
    def num_clusters(self):
        return self.num_groups

    @property
    def num_points(self):
        return self.num_items

    def to_super_index(self) -> torch.Tensor:
        """Return a 1D tensor of indices converting the CSR-formatted
        clustering structure in 'self' into the 'super_index' format.
        """
        # TODO: this assumes 'self.point' is a permutation, shall we
        #  check this (although it requires sorting) ?
        device = self.device
        out = torch.empty((self.num_items,), dtype=torch.long, device=device)
        cluster_idx = torch.arange(self.num_groups, device=device)
        out[self.points] = cluster_idx.repeat_interleave(self.sizes)
        return out

    def select(
            self,
            idx: Union[int, List[int], torch.Tensor, np.ndarray],
            update_sub: bool = True
    ) -> Tuple['Cluster', Tuple[torch.Tensor, torch.Tensor]]:
        """Returns a new Cluster with updated clusters and points, which
        indexes `self` using entries in `idx`. Supports torch and numpy
        fancy indexing. `idx` must NOT contain duplicate entries, as
        this would cause ambiguities in super- and sub- indices.

        NB: if `self` belongs to a NAG, calling this function in
        isolation may break compatibility with point and cluster indices
        in the other hierarchy levels. If consistency matters, prefer
        using NAG indexing instead.

        :parameter
        idx: int or 1D torch.LongTensor or numpy.NDArray
            Cluster indices to select from 'self'. Must NOT contain
            duplicates
        update_sub: bool
            If True, the point (i.e. subpoint) indices will also be
            updated to maintain dense indices. The output will then
            contain '(idx_sub, sub_super)' which can help apply these
            changes to maintain consistency with lower hierarchy levels
            of a NAG.

        :return: cluster, (idx_sub, sub_super)
            clusters: Cluster
                indexed cluster
            idx_sub: torch.LongTensor
                to be used with 'Data.select()' on the sub-level
            sub_super: torch.LongTensor
                to replace 'Data.super_index' on the sub-level
        """
        # Normal CSRData indexing, creates a new object in memory
        cluster = super().select(idx)

        if not update_sub:
            return cluster, (None, None)

        # Convert subpoint indices, in case some subpoints have
        # disappeared. 'idx_sub' is intended to be used with
        # Data.select() on the level below
        # TODO: IMPORTANT consecutive_cluster is a bottleneck for NAG
        #  and Data indexing, can we do better ?
        new_cluster_points, perm = consecutive_cluster(cluster.points)
        idx_sub = cluster.points[perm]
        cluster.points = new_cluster_points

        # Selecting the subpoints with 'idx_sub' will not be
        # enough to maintain consistency with the current points. We
        # also need to update the sub-level's 'Data.super_index', which
        # can be computed from 'cluster'
        sub_super = cluster.to_super_index()

        return cluster, (idx_sub, sub_super)

    def debug(self):
        super().debug()
        assert not has_duplicates(self.points)

    def __repr__(self):
        info = [
            f"{key}={getattr(self, key)}"
            for key in ['num_clusters', 'num_points', 'device']]
        return f"{self.__class__.__name__}({', '.join(info)})"

    @classmethod
    def load(
            cls,
            f: Union[str, h5py.File, h5py.Group],
            idx: Union[int, List, np.ndarray, torch.Tensor] = None,
            update_sub: bool = True,
            verbose: bool = False
    ) -> 'Cluster':
        """Load Cluster from an HDF5 file. See `Cluster.save` for
        writing such file. Options allow reading only part of the
        clusters.

        This reproduces the behavior of Cluster.select but without
        reading the full pointer data from disk.

        :param f: h5 file path of h5py.File or h5py.Group
        :param idx: int, list, numpy.ndarray, torch.Tensor
            Used to select clusters when reading. Supports fancy
            indexing
        :param update_sub: bool
            If True, the point (i.e. subpoint) indices will also be
            updated to maintain dense indices. The output will then
            contain '(idx_sub, sub_super)' which can help apply these
            changes to maintain consistency with lower hierarchy levels
            of a NAG.
        :param verbose: bool

        :return: cluster, (idx_sub, sub_super)
        """
        if not isinstance(f, (h5py.File, h5py.Group)):
            with h5py.File(f, 'r') as file:
                out = cls.load(
                    file, idx=idx, update_sub=update_sub, verbose=verbose)
            return out

        # CSRData load behavior
        out = super().load(f, idx=idx, verbose=verbose)
        cluster = out[0] if isinstance(out, tuple) else out
        
        if not update_sub:
            return cluster, (None, None)
        
        # Convert subpoint indices, in case some subpoints have
        # disappeared. 'idx_sub' is intended to be used with
        # Data.select() on the level below
        # TODO: IMPORTANT consecutive_cluster is a bottleneck for NAG
        #  and Data indexing, can we do better ?
        start = time()
        new_cluster_points, perm = consecutive_cluster(cluster.points)
        idx_sub = cluster.points[perm]
        cluster.points = new_cluster_points
        if verbose:
            print(f'{cls.__name__}.load update_sub     : {time() - start:0.5f}s')

        # Selecting the subpoints with 'idx_sub' will not be
        # enough to maintain consistency with the current points. We
        # also need to update the sublevel's 'Data.super_index', which
        # can be computed from 'cluster'
        start = time()
        sub_super = cluster.to_super_index()
        if verbose:
            print(f'{cls.__name__}.load super_index    : {time() - start:0.5f}s')

        return cluster, (idx_sub, sub_super)


class ClusterBatch(Cluster, CSRBatch):
    """Wrapper for Cluster batching."""

    @classmethod
    def load(
            cls,
            f: Union[str, h5py.File, h5py.Group],
            idx: Union[int, List, np.ndarray, torch.Tensor] = None,
            update_sub: bool = True,
            verbose: bool = False
    ) -> Union['ClusterBatch', 'Cluster']:
        """Load ClusterBatch from an HDF5 file. See `Cluster.save` for
        writing such file. Options allow reading only part of the
        clusters.

        This reproduces the behavior of Cluster.select but without
        reading the full pointer data from disk.

        :param f: h5 file path of h5py.File or h5py.Group
        :param idx: int, list, numpy.ndarray, torch.Tensor
            Used to select clusters when reading. Supports fancy
            indexing
        :param update_sub: bool
            If True, the point (i.e. subpoint) indices will also be
            updated to maintain dense indices. The output will then
            contain '(idx_sub, sub_super)' which can help apply these
            changes to maintain consistency with lower hierarchy levels
            of a NAG.
        :param verbose: bool

        :return: cluster, (idx_sub, sub_super)
        """
        # Indexing breaks batching, so we return a base object if
        # indexing is required
        idx = tensor_idx(idx)
        if idx is not None and idx.shape[0] != 0:
            return cls.get_base_class().load(
                f, idx=idx, update_sub=update_sub, verbose=verbose)

        if not isinstance(f, (h5py.File, h5py.Group)):
            with h5py.File(f, 'r') as file:
                out = cls.load(
                    file, idx=idx, update_sub=update_sub, verbose=verbose)
            return out

        # Check if the file actually corresponds to a batch object
        # rather than its corresponding base object
        if '__sizes__' not in f.keys():
            return cls.get_base_class().load(
                f, idx=idx, update_sub=update_sub, verbose=verbose)

        out = super().load(f, idx=idx, update_sub=update_sub, verbose=verbose)
        out[0].__sizes__ = load_tensor(f['__sizes__'])
        return out
