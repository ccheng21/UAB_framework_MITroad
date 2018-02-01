# '/home/lab/codes_chen/framework_of_github/Results/PatchExtr/road_orgd/chipExtrRegPurge_cSz572x572_pad184'
"""
Created on 01/16/2018
This file show examples of following steps:
    1. Make a collection on inria with RGB data
    2. Modify the GT and map it to (0, 1)
    3. Extract patches of given size
    4. Make train and validation folds
    5. Train a UNet on those patches
"""

import os
import time
import numpy as np
import tensorflow as tf
import uabDataReader
import uabRepoPaths
import uabCrossValMaker
import bohaoCustom.uabPreprocClasses as bPreproc
import uabPreprocClasses
import uab_collectionFunctions
import uab_DataHandlerFunctions
from bohaoCustom import uabMakeNetwork_UNet

# experiment settings
chip_size = (572, 572)
tile_size = (1500, 1500)
batch_size = 5                  # mini-batch size
learn_rate = 1e-4               # learning rate
decay_step = 60                 # learn rate dacay after 60 epochs
decay_rate=0.1                  # learn rate decay to 0.1*before
#########################################################################################################
#epochs=100                      # total number of epochs to rum
epochs=15
#########################################################################################################
start_filter_num=32             # the number of filters at the first layer
n_train = 80                  # number of samples per epoch
n_valid = 10                  # number of samples every validation step
#########################################################################################################
#model_name = 'inria_aug'        # a suffix for model name
model_name = 'road_aug'
#########################################################################################################
#########################################################################################################
#GPU = 1                         # which gpu to use
GPU = None
#########################################################################################################

# make network
# define place holder
X = tf.placeholder(tf.float32, shape=[None, chip_size[0], chip_size[1], 3], name='X')
y = tf.placeholder(tf.int32, shape=[None, chip_size[0], chip_size[1], 1], name='y')
mode = tf.placeholder(tf.bool, name='mode')
model = uabMakeNetwork_UNet.UnetModelCrop({'X':X, 'Y':y},
                                          trainable=mode,
                                          input_size=chip_size,
                                          batch_size=batch_size,
                                          learn_rate=learn_rate,
                                          decay_step=decay_step,
                                          decay_rate=decay_rate,
                                          epochs=epochs,
                                          start_filter_num=start_filter_num)
model.create_graph('X', class_num=2)

# create collection
# the original file is in /ei-edl01/data/uab_datasets/mit_road
blCol = uab_collectionFunctions.uabCollection('road')
opDetObj = bPreproc.uabOperTileDivide(255)          # raod GT has value 0 and 255, we map it back to 0 and 1
############################################################################################
# [4] is the channel id of GT (0&1), [0] is the channel id of GT (0&255)
#rescObj = uabPreprocClasses.uabPreprocMultChanOp([], 'GT_Divide.tif', 'Map GT to (0, 1)', [3], opDetObj)
rescObj = uabPreprocClasses.uabPreprocMultChanOp([], 'GT_Divide.tif', 'Map GT to (0, 1)', [0], opDetObj)
############################################################################################
rescObj.run(blCol)
print(blCol.readMetadata())                         # now road collection has 4 channels, the last one is GT with (0, 1)
blCol.getMetaDataInfo([1, 2, 3], class_info='background,road', forcerun=False)
img_mean = blCol.getChannelMeans([1, 2, 3])         # get mean of rgb info

# extract patches
extrObj = uab_DataHandlerFunctions.uabPatchExtr([1, 2, 3, 4], # extract all 4 channels
                                                cSize=chip_size, # patch size as 572*572
                                                name='RegPurge',
                                                numPixOverlap=int(model.get_overlap()/2),  # overlap as 92
                                                extSave=['jpg', 'jpg', 'jpg', 'png'], # save rgb files as jpg and gt as png
                                                isTrain=True,
                                                gtInd=3,
                                                pad=model.get_overlap()) # pad around the tiles
patchDir = extrObj.run(blCol)

# make data reader
chipFiles = os.path.join(patchDir, 'fileList.txt')
# use uabCrossValMaker to get fileLists for training and validation
############################################################################################
# Done: 'force_tile' -> 'city'
# 'fileList.txt' is in "/home/lab/codes_chen/framework_of_github/Results/PatchExtr/road_orgd/chipExtrRegPurge_cSz572x572_pad184"
#idx, file_list = uabCrossValMaker.uabUtilGetFolds(patchDir, 'fileList.txt', 'force_tile')
idx, file_list = uabCrossValMaker.uabUtilGetFolds(patchDir, 'fileList.txt', 'city')

############################################################################################
# Done: change idx(key) from force_tile to city: [0-37] -> [0-2], and identify the key of training: test(print) the idx of 0,1,2 seperately to find the training idx and see whether all training patchs have the same key
# This is cross-validation
# idx[0] = 'test', idx[1] = 'training', idx[2] = 'validation'
# use first 5 tiles for validation
#file_list_train = uabCrossValMaker.make_file_list_by_key(idx, file_list, [i for i in range(6, 37)])
#file_list_valid = uabCrossValMaker.make_file_list_by_key(idx, file_list, [i for i in range(0, 6)])
file_list_train = uabCrossValMaker.make_file_list_by_key(idx, file_list, 1)
file_list_valid = uabCrossValMaker.make_file_list_by_key(idx, file_list, 2)
############################################################################################

with tf.name_scope('image_loader'):
    # GT has no mean to subtract, append a 0 for block mean
    dataReader_train = uabDataReader.ImageLabelReader([3], [0, 1, 2], patchDir, file_list_train, chip_size, tile_size,
                                                      batch_size, dataAug='flip,rotate', block_mean=np.append([0], img_mean))
    # no augmentation needed for validation
    dataReader_valid = uabDataReader.ImageLabelReader([3], [0, 1, 2], patchDir, file_list_valid, chip_size, tile_size,
                                                      batch_size, dataAug=' ', block_mean=np.append([0], img_mean))

# train
start_time = time.time()

model.train_config('X', 'Y', n_train, n_valid, chip_size, uabRepoPaths.modelPath, loss_type='xent')
# Where can I input the dir name of real training & validation dataset? ANS: Automatic search
model.run(train_reader=dataReader_train,
          valid_reader=dataReader_valid,
          pretrained_model_dir=None,        # train from scratch, no need to load pre-trained model
          isTrain=True,
          img_mean=img_mean,
          verb_step=100,                    # print a message every 100 step(sample)
          save_epoch=5,                     # save the model every 5 epochs
          gpu=GPU,
          tile_size=tile_size,
          patch_size=chip_size)

duration = time.time() - start_time
print('duration {:.2f} hours'.format(duration/60/60))

