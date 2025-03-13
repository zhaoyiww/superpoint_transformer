# ---------------------------------------------------------------------------- #
# just run once, end-to-end to get the three levels of superpoint partition results
# input root, config file
# processing mattertal ROIs as an example
# ---------------------------------------------------------------------------- #

import hydra
from src.utils import init_config
from easydict import EasyDict as edict
from src.transforms import *
import shutil
from tqdm import tqdm
import os
import os.path as osp
import numpy as np
import torch
from colorhash import ColorHash
import open3d as o3d

# superpoint transformer/src/utils/color.py
def int_to_plotly_rgb(x):
    """Convert 1D torch.Tensor of int into plotly-friendly RGB format.
    This operation is deterministic on the int values.
    """
    assert isinstance(x, torch.Tensor)
    assert x.dim() == 1
    assert not x.is_floating_point()
    x = x.cpu().long().numpy()
    palette = np.array([ColorHash(i).rgb for i in range(x.max() + 1)])
    return palette[x]

############
# pretrained model loading, only once

# sematic, test, dales_11g, important for config hyper-parameters
device_widget = 'cuda:0'
# task_widget = 'panoptic'
task_widget = 'semantic'
split_widget = 'test'
# experiment = 's3dis_11g'
# light version compared to dales_32g, limited by the GPU memory
experiment = 'dales_11g'
# not used in this script
pretrained_weight = '/scratch2/zhawang/projects/deformation/DeformHD_local/weights/spt-2_dales.ckpt'

#################
# initialize configs
# load corresponding 'yaml' from 'configs/experiment/semantic/dales_11g.yaml'
cfg = init_config(overrides=[
    f"experiment={task_widget}/{experiment}",
    f"ckpt_path={pretrained_weight}",
    f"datamodule.load_full_res_idx={True}"  # only when you need full-resolution predictions
])

# should write some changes of config parameters in the following
# partition feature types, either xyz or xyzrgb
# xyzrgb, xyz
partition_type = 'xyzrgb'
# partition high-level features
if partition_type == 'xyzrgb':
    cfg.datamodule.partition_hf = ['linearity', 'planarity', 'scattering', 'intensity']
elif partition_type == 'xyz':
    cfg.datamodule.partition_hf = ['linearity', 'planarity', 'scattering']
else:
    raise ValueError(f'partition_type {partition_type} not recognized')
# print(cfg)
#################

#################
# edit external config file
config = edict(dict())
config.data_dir = '/scratch2/zhawang/projects/deformation/DeformHD_local/output/demo_offline'
config.file_folder = 'tiled_data/overlap'
config.tile_id = '1'
#################

#################
# load data
# Instantiate the datamodule
datamodule = hydra.utils.instantiate(cfg.datamodule)
datamodule.prepare_data(config)
datamodule.setup(config)
#################

# Pick among train, val, and test datasets. It is important to note that
# the train dataset produces augmented spherical samples of large
# scenes, while the val and test dataset load entire tiles at once
dataset = datamodule.test_dataset
# print(dataset)
# Print a summary of the datasets' classes
# dataset.print_classes()

# # Instantiate the model
# model = hydra.utils.instantiate(cfg.model)
# # print(model)
# # Move model to selected device
# model = model.eval().to(device_widget)

# For the sake of visualization, we require that NAGAddKeysTo does not
# remove input Data attributes after moving them to Data.x, so we may
# visualize them
for t in dataset.on_device_transform.transforms:
    if isinstance(t, NAGAddKeysTo):
        t.delete_after = False


# Load the first dataset item. This will return the hierarchical
# partition of an entire tile, as a NAG object
# pt_path = osp.join(config.data_dir, 'viz_results')
# pt_path = '/scratch2/zhawang/projects/deformation/DeformHD_local/output/Brienz/AOI_2/spt_interim/viz_results/'
# shutil.rmtree(path=pt_path, ignore_errors=True)
for i in range(len(dataset)):
    nag = dataset[i]

    # Apply on-device transforms on the NAG object. For the train dataset,
    # this will select a spherical sample of the larger tile and apply some
    # data augmentations. For the validation and test datasets, this will
    # prepare an entire tile for inference
    nag = dataset.on_device_transform(nag.to(device_widget))

    # output superpoint indices
    indices_raw_to_level0 = nag[0].sub.to_super_index()
    indices_level0_to_level1 = nag[0].super_index
    indices_level1_to_level2 = nag[1].super_index
    indices_level2_to_level3 = nag[2].super_index

    if i == 0:
        pts = o3d.io.read_point_cloud(osp.join(config.data_dir, config.file_folder, f'source_tile_{config.tile_id}.ply'))
    elif i == 1:
        pts = o3d.io.read_point_cloud(osp.join(config.data_dir, config.file_folder, f'target_tile_{config.tile_id}.ply'))
    else:
        raise ValueError('Only two tiles are available!')
    pts = np.asarray(pts.points)

    # a = indices_raw_to_level0.cpu()
    colors_0 = int_to_plotly_rgb(indices_level0_to_level1)
    colors_1 = int_to_plotly_rgb(indices_level1_to_level2)
    colors_2 = int_to_plotly_rgb(indices_level2_to_level3)
    # colors_3 = int_to_plotly_rgb(torch.arange(nag[3]['pos'].shape[0]))

    indices_raw_to_level1 = torch.stack([indices_level0_to_level1[int(j)] for j in indices_raw_to_level0])
    indices_raw_to_level2 = torch.stack(
        [indices_level1_to_level2[int(indices_level0_to_level1[int(j)])] for j in indices_raw_to_level0])
    # indices_raw_to_level3 = torch.stack(
    #     [indices_level2_to_level3[int(indices_level1_to_level2[int(indices_level0_to_level1[int(j)])])] for j in
    #      indices_raw_to_level0])

    colors_raw_to_level0 = colors_0[indices_raw_to_level0.cpu(), :]
    colors_raw_to_level1 = colors_1[indices_raw_to_level1.cpu(), :]
    colors_raw_to_level2 = colors_2[indices_raw_to_level2.cpu(), :]
    # colors_raw_to_level3 = colors_3[indices_raw_to_level3.cpu(), :]

    # optional, output individual layers
    output_path = osp.join(config.data_dir, 'superpoint_partition')
    os.makedirs(output_path, exist_ok=True)
    if i == 0:
        np.savetxt(osp.join(output_path, 'source_partition_three_levels.txt'),
                   np.concatenate([pts, colors_raw_to_level0, indices_raw_to_level0.cpu()[:, None],
                                   colors_raw_to_level1, indices_raw_to_level1.cpu()[:, None],
                                   colors_raw_to_level2, indices_raw_to_level2.cpu()[:, None]], axis=1),
                   fmt='%.3f %.3f %.3f %d %d %d %d %d %d %d %d %d %d %d %d')
        # optional, output individual layers
        np.savetxt(osp.join(output_path, 'source_partition_level0.txt'),
                   np.concatenate([pts, colors_raw_to_level0, indices_raw_to_level0.cpu()[:, None]], axis=1),
                   fmt='%.3f %.3f %.3f %d %d %d %d')
        np.savetxt(osp.join(output_path, 'source_partition_level1.txt'),
                   np.concatenate([pts, colors_raw_to_level1, indices_raw_to_level1.cpu()[:, None]], axis=1),
                   fmt='%.3f %.3f %.3f %d %d %d %d')
        np.savetxt(osp.join(output_path, 'source_partition_level2.txt'),
                   np.concatenate([pts, colors_raw_to_level2, indices_raw_to_level2.cpu()[:, None]], axis=1),
                   fmt='%.3f %.3f %.3f %d %d %d %d')
        # np.savetxt(osp.join(output_path, 'source_partition_level3.txt'),
        #            np.concatenate([pts, colors_raw_to_level3, indices_raw_to_level3.cpu()[:, None]], axis=1),
        #            fmt='%.3f %.3f %.3f %d %d %d %d')
    elif i == 1:
        np.savetxt(osp.join(output_path, 'target_partition_three_levels.txt'),
                   np.concatenate([pts, colors_raw_to_level0, indices_raw_to_level0.cpu()[:, None],
                                   colors_raw_to_level1, indices_raw_to_level1.cpu()[:, None],
                                   colors_raw_to_level2, indices_raw_to_level2.cpu()[:, None]], axis=1),
                   fmt='%.3f %.3f %.3f %d %d %d %d %d %d %d %d %d %d %d %d')
        # optional, output individual layers
        np.savetxt(osp.join(output_path, 'target_partition_level0.txt'),
                   np.concatenate([pts, colors_raw_to_level0, indices_raw_to_level0.cpu()[:, None]], axis=1),
                   fmt='%.3f %.3f %.3f %d %d %d %d')
        np.savetxt(osp.join(output_path, 'target_partition_level1.txt'),
                   np.concatenate([pts, colors_raw_to_level1, indices_raw_to_level1.cpu()[:, None]], axis=1),
                   fmt='%.3f %.3f %.3f %d %d %d %d')
        np.savetxt(osp.join(output_path, 'target_partition_level2.txt'),
                   np.concatenate([pts, colors_raw_to_level2, indices_raw_to_level2.cpu()[:, None]], axis=1),
                   fmt='%.3f %.3f %.3f %d %d %d %d')
        # np.savetxt(osp.join(output_path, 'target_partition_level3.txt'),
        #            np.concatenate([pts, colors_raw_to_level3, indices_raw_to_level3.cpu()[:, None]], axis=1),
        #            fmt='%.3f %.3f %.3f %d %d %d %d')
    b = 0




