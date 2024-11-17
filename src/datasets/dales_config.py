import numpy as np


########################################################################
#                         Download information                         #
########################################################################

FORM_URL = 'https://docs.google.com/forms/d/e/1FAIpQLSefhHMMvN0Uwjnj_vWQgYSvtFOtaoGFWsTIcRuBTnP09NHR7A/viewform?fbzx=5530674395784263977'

# DALES in LAS format
LAS_TAR_NAME = 'dales_semantic_segmentation_las.tar.gz'
LAS_UNTAR_NAME = "dales_las"

# DALES in PLY format
PLY_TAR_NAME = 'dales_semantic_segmentation_ply.tar.gz'
PLY_UNTAR_NAME = "dales_ply"

# DALES in PLY, only version with intensity and instance labels
OBJECTS_TAR_NAME = 'DALESObjects.tar.gz'
OBJECTS_UNTAR_NAME = "DALESObjects"


########################################################################
#                              Data splits                             #
########################################################################

# The validation set was arbitrarily chosen as the x last train tiles:
TILES = {
    'train': [
    ],

    'val': [
        # '5145_54340_new',
        # '5095_54455_new',
        # '5110_54475_new'
    ],

    'test': [
        'source_tile_0_overlap',
    ]}


########################################################################
#                                Labels                                #
########################################################################

DALES_NUM_CLASSES = 8

ID2TRAINID = np.asarray([8, 0, 1, 2, 3, 4, 5, 6, 7])

CLASS_NAMES = [
    'Ground',
    'Vegetation',
    'Cars',
    'Trucks',
    'Power lines',
    'Fences',
    'Poles',
    'Buildings',
    'Unknown']

CLASS_COLORS = np.asarray([
    [243, 214, 171],  # sunset
    [ 70, 115,  66],  # fern green
    [233,  50, 239],
    [243, 238,   0],
    [190, 153, 153],
    [  0, 233,  11],
    [239, 114,   0],
    [214,   66,  54],  # vermillon
    [  0,   8, 116]])

# For instance segmentation
MIN_OBJECT_SIZE = 100
THING_CLASSES = [2, 3, 4, 5, 6, 7]
STUFF_CLASSES = [i for i in range(DALES_NUM_CLASSES) if not i in THING_CLASSES]
