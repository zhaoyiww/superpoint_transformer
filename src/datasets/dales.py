import os
import sys
import torch
import shutil
import logging
import os.path as osp
from plyfile import PlyData
from typing import List
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.data import extract_tar

from ..datasets import BaseDataset
from ..data import Data, InstanceData
from ..datasets.dales_config import *


DIR = os.path.dirname(os.path.realpath(__file__))
log = logging.getLogger(__name__)


# Occasional Dataloader issues with DALES on some machines. Hack to
# solve this:
# https://stackoverflow.com/questions/73125231/pytorch-dataloaders-bad-file-descriptor-and-eof-for-workers0
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


__all__ = ['DALES', 'MiniDALES']


########################################################################
#                                 Utils                                #
########################################################################

def read_dales_tile(
        filepath: str,
        xyz: bool = True,
        intensity: bool = True,
        semantic: bool = True,
        instance: bool = True,
        remap: bool = False
) -> Data:
    """Read a DALES tile saved as PLY.

    :param filepath: str
        Absolute path to the PLY file
    :param xyz: bool
        Whether XYZ coordinates should be saved in the output Data.pos
    :param intensity: bool
        Whether intensity should be saved in the output Data.intensity
    :param semantic: bool
        Whether semantic labels should be saved in the output Data.y
    :param instance: bool
        Whether instance labels should be saved in the output Data.obj
    :param remap: bool
        Whether semantic labels should be mapped from their DALES ID
        to their train ID.
    """
    data = Data()
    # key = 'testing'
    key = 'vertex'
    with open(filepath, "rb") as f:
        tile = PlyData.read(f)

        # if xyz:
        #     pos = torch.stack([
        #         torch.FloatTensor(tile[key][axis])
        #         for axis in ["x", "y", "z"]], dim=-1)
        #     pos_offset = pos[0]
        #     data.pos = pos - pos_offset
        #     data.pos_offset = pos_offset

        if xyz:
            pos = torch.stack([
                torch.FloatTensor(np.ascontiguousarray(tile[key][axis]))
                for axis in ["x", "y", "z"]], dim=-1)
            pos_offset = pos[0]
            data.pos = pos - pos_offset
            data.pos_offset = pos_offset

        if intensity:
            # Heuristic to bring the intensity distribution in [0, 1]
            # data.intensity = torch.FloatTensor(
            #     tile[key]['intensity']).clip(min=0, max=60000) / 60000

            # use RGB values to compute grayscale values for intensity
            vertex_properties = tile['vertex'].properties
            property_names = [prop.name for prop in vertex_properties]
            if 'red' in property_names:
                rgb = torch.stack([
                    torch.FloatTensor(tile[key][axis])
                    for axis in ["red", "green", "blue"]], dim=-1)

                weights = torch.tensor([0.2989, 0.5870, 0.1140], dtype=rgb.dtype, device=rgb.device)
                data.intensity = torch.matmul(rgb, weights)
                data.intensity = (data.intensity - 0) / (255 - 0)
            else:
                data.intensity = torch.FloatTensor(
                    tile[key]['intensity'])
                data.intensity = (data.intensity - data.intensity.min()) / (data.intensity.max() - data.intensity.min())

        if semantic:
            y = torch.LongTensor(tile[key]['sem_class'])
            data.y = torch.from_numpy(ID2TRAINID)[y] if remap else y

        if instance:
            idx = torch.arange(data.num_points)
            obj = torch.LongTensor(tile[key]['ins_class'])
            obj = consecutive_cluster(obj)[0]
            count = torch.ones_like(obj)
            y = torch.LongTensor(tile[key]['sem_class'])
            y = torch.from_numpy(ID2TRAINID)[y] if remap else y
            data.obj = InstanceData(idx, obj, count, y, dense=True)

    return data


########################################################################
#                                DALES                                 #
########################################################################

class DALES(BaseDataset):
    """DALES dataset.

    Dataset website: https://udayton.edu/engineering/research/centers/vision_lab/research/was_data_analysis_and_processing/dale.php

    Parameters
    ----------
    root : `str`
        Root directory where the dataset should be saved.
    stage : {'train', 'val', 'test', 'trainval'}
    transform : `callable`
        transform function operating on data.
    pre_transform : `callable`
        pre_transform function operating on data.
    pre_filter : `callable`
        pre_filter function operating on data.
    on_device_transform: `callable`
        on_device_transform function operating on data, in the
        'on_after_batch_transfer' hook. This is where GPU-based
        augmentations should be, as well as any Transform you do not
        want to run in CPU-based DataLoaders
    """

    _form_url = FORM_URL
    _zip_name = OBJECTS_TAR_NAME
    _las_name = LAS_TAR_NAME
    _ply_name = PLY_TAR_NAME
    _unzip_name = OBJECTS_UNTAR_NAME

    @property
    def class_names(self) -> List[str]:
        """List of string names for dataset classes. This list must be
        one-item larger than `self.num_classes`, with the last label
        corresponding to 'void', 'unlabelled', 'ignored' classes,
        indicated as `y=self.num_classes` in the dataset labels.
        """
        return CLASS_NAMES

    @property
    def num_classes(self) -> int:
        """Number of classes in the dataset. Must be one-item smaller
        than `self.class_names`, to account for the last class name
        being used for 'void', 'unlabelled', 'ignored' classes,
        indicated as `y=self.num_classes` in the dataset labels.
        """
        return DALES_NUM_CLASSES

    @property
    def stuff_classes(self) -> List[int]:
        """List of 'stuff' labels for INSTANCE and PANOPTIC
        SEGMENTATION (setting this is NOT REQUIRED FOR SEMANTIC
        SEGMENTATION alone). By definition, 'stuff' labels are labels in
        `[0, self.num_classes-1]` which are not 'thing' labels.

        In instance segmentation, 'stuff' classes are not taken into
        account in performance metrics computation.

        In panoptic segmentation, 'stuff' classes are taken into account
        in performance metrics computation. Besides, each cloud/scene
        can only have at most one instance of each 'stuff' class.

        IMPORTANT:
        By convention, we assume `y ∈ [0, self.num_classes-1]` ARE ALL
        VALID LABELS (i.e. not 'ignored', 'void', 'unknown', etc), while
        `y < 0` AND `y >= self.num_classes` ARE VOID LABELS.
        """
        return STUFF_CLASSES

    @property
    def class_colors(self) -> List[List[int]]:
        """Colors for visualization, if not None, must have the same
        length as `self.num_classes`. If None, the visualizer will use
        the label values in the data to generate random colors.
        """
        return CLASS_COLORS

    @property
    def all_base_cloud_ids(self) -> List[str]:
        """Dictionary holding lists of paths to the clouds, for each
        stage.

        The following structure is expected:
            `{'train': [...], 'val': [...], 'test': [...]}`
        """
        return TILES

    def download_dataset(self) -> None:
        """Download the DALES Objects dataset.
        """
        # Manually download the dataset
        if not osp.exists(osp.join(self.root, self._zip_name)):
            log.error(
                f"\nDALES does not support automatic download.\n"
                f"Please, register yourself by filling up the form at "
                f"{self._form_url}\n"
                f"From there, manually download the '{self._zip_name}' into "
                f"your '{self.root}/' directory and re-run.\n"
                f"The dataset will automatically be unzipped into the "
                f"following structure:\n"
                f"{self.raw_file_structure}\n"
                f"⛔ Make sure you DO NOT download the "
                f"'{self._las_name}' nor '{self._ply_name}' versions, which "
                f"do not contain all required point attributes.\n")
            sys.exit(1)

        # Unzip the file and rename it into the `root/raw/` directory
        extract_tar(osp.join(self.root, self._zip_name), self.root)
        shutil.rmtree(self.raw_dir)
        os.rename(osp.join(self.root, self._unzip_name), self.raw_dir)

    def read_single_raw_cloud(self, raw_cloud_path: str) -> 'Data':
        """Read a single raw cloud and return a `Data` object, ready to
        be passed to `self.pre_transform`.

        This `Data` object should contain the following attributes:
          - `pos`: point coordinates
          - `y`: OPTIONAL point semantic label
          - `obj`: OPTIONAL `InstanceData` object with instance labels
          - `rgb`: OPTIONAL point color
          - `intensity`: OPTIONAL point LiDAR intensity

        IMPORTANT:
        By convention, we assume `y ∈ [0, self.num_classes-1]` ARE ALL
        VALID LABELS (i.e. not 'ignored', 'void', 'unknown', etc),
        while `y < 0` AND `y >= self.num_classes` ARE VOID LABELS.
        This applies to both `Data.y` and `Data.obj.y`.
        """
        return read_dales_tile(
            raw_cloud_path, intensity=True, semantic=False, instance=False,
            remap=True)

    @property
    def raw_file_structure(self) -> str:
        return f"""
    {self.root}/
        └── raw/
            └── {{train, test}}/
                └── {{tile_name}}.ply
            """

    def id_to_relative_raw_path(self, id: str) -> str:
        """Given a cloud id as stored in `self.cloud_ids`, return the
        path (relative to `self.raw_dir`) of the corresponding raw
        cloud.
        """
        if id in self.all_cloud_ids['train']:
            stage = 'train'
        elif id in self.all_cloud_ids['val']:
            stage = 'train'
        elif id in self.all_cloud_ids['test']:
            stage = 'test'
        else:
            raise ValueError(f"Unknown tile id '{id}'")
        return osp.join(stage, self.id_to_base_id(id) + '.ply')

    def processed_to_raw_path(self, processed_path: str) -> str:
        """Return the raw cloud path corresponding to the input
        processed path.
        """
        # Extract useful information from <path>
        stage, hash_dir, cloud_id = \
            osp.splitext(processed_path)[0].split(os.sep)[-3:]

        # Raw 'val' and 'trainval' tiles are all located in the
        # 'raw/train/' directory
        stage = 'train' if stage in ['trainval', 'val'] else stage

        # Remove the tiling in the cloud_id, if any
        base_cloud_id = self.id_to_base_id(cloud_id)

        # Read the raw cloud data
        raw_path = osp.join(self.raw_dir, stage, base_cloud_id + '.ply')

        return raw_path


########################################################################
#                              MiniDALES                               #
########################################################################

class MiniDALES(DALES):
    """A mini version of DALES with only a few windows for
    experimentation.
    """
    _NUM_MINI = 2

    @property
    def all_cloud_ids(self) -> List[str]:
        return {k: v[:self._NUM_MINI] for k, v in super().all_cloud_ids.items()}

    @property
    def data_subdir_name(self) -> str:
        return self.__class__.__bases__[0].__name__.lower()

    # We have to include this method, otherwise the parent class skips
    # processing
    def process(self) -> None:
        super().process()

    # We have to include this method, otherwise the parent class skips
    # processing
    def download(self) -> None:
        super().download()
