import os
import sys
import glob
import torch
import shutil
import logging
from zipfile import ZipFile
from plyfile import PlyData
from typing import List
from torch_geometric.data.extract import extract_zip
from torch_geometric.nn.pool.consecutive import consecutive_cluster

from src.datasets import BaseDataset
from src.data import Data, InstanceData
from src.datasets.kitti360_config import *
from src.utils.neighbors import knn_2
from src.utils.color import to_float_rgb


DIR = os.path.dirname(os.path.realpath(__file__))
log = logging.getLogger(__name__)


# Occasional Dataloader issues with KITTI360 on some machines. Hack to
# solve this:
# https://stackoverflow.com/questions/73125231/pytorch-dataloaders-bad-file-descriptor-and-eof-for-workers0
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


__all__ = ['KITTI360', 'MiniKITTI360']


########################################################################
#                                 Utils                                #
########################################################################

def read_kitti360_window(
        filepath: str,
        xyz: bool = True,
        rgb: bool = True,
        semantic: bool = True,
        instance: bool = True,
        remap: bool = False
) -> Data:
    """Read a KITTI-360 window –i.e. a tile– saved as PLY.

    :param filepath: str
        Absolute path to the PLY file
    :param xyz: bool
        Whether XYZ coordinates should be saved in the output Data.pos
    :param rgb: bool
        Whether RGB colors should be saved in the output Data.rgb
    :param semantic: bool
        Whether semantic labels should be saved in the output Data.y
    :param instance: bool
        Whether instance labels should be saved in the output Data.obj
    :param remap: bool
        Whether semantic labels should be mapped from their KITTI-360 ID
        to their train ID. For more details, see:
        https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/evaluation/semantic_3d/evalPointLevelSemanticLabeling.py
    """
    data = Data()
    with open(filepath, "rb") as f:
        window = PlyData.read(f)
        attributes = [p.name for p in window['vertex'].properties]

        if xyz:
            pos = torch.stack([
                torch.FloatTensor(window["vertex"][axis])
                for axis in ["x", "y", "z"]], dim=-1)
            pos_offset = pos[0]
            data.pos = pos - pos_offset
            data.pos_offset = pos_offset

        if rgb:
            data.rgb = to_float_rgb(torch.stack([
                torch.FloatTensor(window["vertex"][axis])
                for axis in ["red", "green", "blue"]], dim=-1))

        if semantic and 'semantic' in attributes:
            y = torch.LongTensor(window["vertex"]['semantic'])
            data.y = torch.from_numpy(ID2TRAINID)[y] if remap else y

        if instance and 'instance' in attributes:
            idx = torch.arange(data.num_points)
            obj = torch.LongTensor(window["vertex"]['instance'])
            # is_stuff = obj % 1000 == 0
            # obj[is_stuff] = 0
            obj = consecutive_cluster(obj)[0]
            count = torch.ones_like(obj)
            y = torch.LongTensor(window["vertex"]['semantic'])
            y = torch.from_numpy(ID2TRAINID)[y] if remap else y
            data.obj = InstanceData(idx, obj, count, y, dense=True)

    return data


########################################################################
#                               KITTI360                               #
########################################################################

class KITTI360(BaseDataset):
    """KITTI360 dataset.

    Dataset website: http://www.cvlibs.net/datasets/kitti-360/

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

    _form_url = CVLIBS_URL
    _trainval_zip_name = DATA_3D_SEMANTICS_ZIP_NAME
    _test_zip_name = DATA_3D_SEMANTICS_TEST_ZIP_NAME
    _unzip_name = UNZIP_NAME

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
        return KITTI360_NUM_CLASSES

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
        return WINDOWS

    def download_dataset(self) -> None:
        """Download the KITTI-360 dataset.
        """
        # Name of the downloaded dataset zip
        zip_name = self._test_zip_name if self.stage == 'test' \
            else self._trainval_zip_name

        # Accumulated 3D point clouds with annotations
        if not osp.exists(osp.join(self.root, zip_name)):
            if self.stage != 'test':
                msg = 'Accumulated Point Clouds for Train & Val (12G)'
            else:
                msg = 'Accumulated Point Clouds for Test (1.2G)'
            log.error(
                f"\nKITTI-360 does not support automatic download.\n"
                f"Please go to the official webpage {self._form_url}, "
                f"manually download the '{msg}' (i.e. '{zip_name}') to your "
                f"'{self.root}/' directory, and re-run.\n"
                f"The dataset will automatically be unzipped into the "
                f"following structure:\n"
                f"{self.raw_file_structure}\n")
            sys.exit(1)

        # Unzip the file and place its content into the expected data
        # structure inside `root/raw/` directory
        extract_zip(osp.join(self.root, zip_name), self.raw_dir)
        stage = 'test' if self.stage == 'test' else 'train'
        seqs = os.listdir(osp.join(self.raw_dir, 'data_3d_semantics', stage))
        for seq in seqs:
            source = osp.join(self.raw_dir, 'data_3d_semantics', stage, seq)
            target = osp.join(self.raw_dir, 'data_3d_semantics', seq)
            shutil.move(source, target)
        shutil.rmtree(osp.join(self.raw_dir, 'data_3d_semantics', stage))

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
        return read_kitti360_window(
            raw_cloud_path, semantic=True, instance=True, remap=True)

    @property
    def raw_file_structure(self) -> str:
        return f"""
    {self.root}/
        └── raw/
            └── data_3d_semantics/
                └── 2013_05_28_drive_{{seq:0>4}}_sync/
                    └── static/
                        └── {{start_frame:0>10}}_{{end_frame:0>10}}.ply
            """

    def id_to_relative_raw_path(self, id: str) -> str:
        """Given a cloud id as stored in `self.cloud_ids`, return the
        path (relative to `self.raw_dir`) of the corresponding raw
        cloud.
        """
        id = self.id_to_base_id(id)
        return osp.join(
            'data_3d_semantics', id.split(os.sep)[0], 'static',
            id.split(os.sep)[1] + '.ply')

    def processed_to_raw_path(self, processed_path: str) -> str:
        """Return the raw cloud path corresponding to the input
        processed path.
        """
        # Extract useful information from <path>
        stage, hash_dir, sequence_name, cloud_id = \
            osp.splitext(processed_path)[0].split(os.sep)[-4:]

        # Remove the tiling in the cloud_id, if any
        base_cloud_id = self.id_to_base_id(cloud_id)

        # Read the raw cloud data
        raw_path = osp.join(
            self.raw_dir, 'data_3d_semantics', sequence_name, 'static',
            base_cloud_id + '.ply')

        return raw_path

    def make_submission(
            self,
            idx: int,
            pred: torch.Tensor,
            pos: torch.Tensor,
            submission_dir: str = None
    ) -> None:
        """Prepare data for a sumbission to KITTI360 for 3D semantic
        Segmentation on the test set.

        Expected submission format is detailed here:
        https://github.com/autonomousvision/kitti360Scripts/tree/master/kitti360scripts/evaluation/semantic_3d
        """
        if self.xy_tiling or self.pc_tiling:
            raise NotImplementedError(
                f"Submission generation not implemented for tiled KITTI360 "
                f"datasets yet...")

        # Make sure the prediction is a 1D tensor
        if pred.dim() != 1:
            raise ValueError(
                f'The submission predictions must be 1D tensors, '
                f'received {type(pred)} of shape {pred.shape} instead.')

        # TODO:
        #  - handle tiling
        #  - handle geometric transformations of test data, shuffling of points and of tiles in the dataloader
        #  - handle multiple tiles in the dataloader...
        # Initialize the submission directory
        submission_dir = submission_dir or self.submission_dir
        if not osp.exists(submission_dir):
            os.makedirs(submission_dir)

        # Read the raw point cloud
        raw_path = osp.join(
            self.raw_dir, self.id_to_relative_raw_path(self.cloud_ids[idx]))
        data_raw = self.sanitized_read_single_raw_cloud(raw_path)

        # Search the nearest neighbor of each point and apply the
        # neighbor's class to the points
        neighbors = knn_2(pos, data_raw.pos, 1, r_max=1)[0]
        pred_raw = pred[neighbors]

        # Map TrainId labels to expected Ids
        pred_raw = np.asarray(pred_raw)
        pred_remapped = TRAINID2ID[pred_raw].astype(np.uint8)

        # Recover sequence and window information from stage dataset's
        # windows and format those to match the expected file name:
        # {seq:0>4}_{start_frame:0>10}_{end_frame:0>10}.npy
        sequence_name, window_name = self.id_to_base_id(
            self.cloud_ids[idx]).split(os.sep)
        seq = sequence_name.split('_')[-2]
        start_frame, end_frame = window_name.split('_')
        filename = f'{seq:0>4}_{start_frame:0>10}_{end_frame:0>10}.npy'

        # Save the window submission
        np.save(osp.join(submission_dir, filename), pred_remapped)

    def finalize_submission(self, submission_dir: str) -> None:
        """This should be called once all window submission files have
        been saved using `self._make_submission`. This will zip them
        together as expected by the KITTI360 submission server.
        """
        zipObj = ZipFile(f'{submission_dir}.zip', 'w')
        for p in glob.glob(osp.join(submission_dir, '*.npy')):
            zipObj.write(p)
        zipObj.close()


########################################################################
#                             MiniKITTI360                             #
########################################################################

class MiniKITTI360(KITTI360):
    """A mini version of KITTI360 with only a few windows for
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
