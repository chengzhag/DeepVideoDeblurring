import glob
import os
import math
import re
import random
import imageio
import numpy as np
import time
from PIL import Image
import tensorflow as tf
from io import BytesIO

# Parameters
alignments = ['_nowarp','_OF','_homography']
inputDir = '/home/omnisky/Desktop/zc/DeepVideoDeblurring/data'
inputFolderPrefix = 'training_real_all_nostab'
gtDir = '/home/omnisky/Desktop/zc/DeepVideoDeblurring/dataset/quantitative_datasets'

saveDir = '../data'
saveFolderPrefixTrain = 'training_augumented_all_nostab'
saveFolderPrefixValid = 'validating_augumented_all_nostab'

frameExt = '.jpg'
cropWidth = 128
nCrop = 10
nArguments = 2 * 4 * 4
widthNeighbor = 5
batchSize = 64
currSpeed = 0
argumentZoom = [1 / 4, 1 / 3, 1 / 2, 1]

# Scan videos
validset = ['IMG_0030', 'IMG_0049', 'IMG_0021', '720p_240fps_2', 'IMG_0032',
            'IMG_0033', 'IMG_0031', 'IMG_0003', 'IMG_0039', 'IMG_0037']

inputFolders = []
print('Input folders: ')
for alignment in alignments:
    inputFolders.append(os.path.join(inputDir, inputFolderPrefix + alignment))
    print('\t' + inputFolders[-1])

videoNames = []
for inputFolder in inputFolders:
    folders = os.listdir(inputFolder)
    folders.sort()
    # for videoFolder in folders:
    #     print(videoFolder)
    videoNames.append(folders)


# Generate batches
def scanFrames(inputFolder, gtDir, videoNames):
    print('Scanning frames...')
    frameDirs = []
    for iVideo in range(len(videoNames)):
        videoName = videoNames[iVideo]
        inputFrameFolder = os.path.join(inputFolder, videoName)
        inputFrameDirs = glob.glob(os.path.join(inputFrameFolder, 'image_0', '*' + frameExt))
        inputFrameDirs.sort()
        inputFrameNames = [os.path.split(inputFrameDir)[1] for inputFrameDir in inputFrameDirs]
        GTFrameFolder = os.path.join(gtDir, videoName, 'GT')
        GTFrameDirs = glob.glob(os.path.join(GTFrameFolder, '*' + frameExt))
        GTFrameDirs.sort()
        for iFrame in range(len(inputFrameDirs)):
            neighborDirs = []
            for iNeighbor in range(widthNeighbor):
                iFrameNeighbor = iNeighbor - math.floor(widthNeighbor / 2)
                neighborDirs.append(os.path.join(
                    inputFrameFolder,
                    'image_%d' % iFrameNeighbor,
                    inputFrameNames[iFrame]
                ))
            frameDirs.append([neighborDirs, GTFrameDirs[iFrame]])
        print('Found %d frames, video %d/%d' % (len(frameDirs), iVideo + 1, len(videoNames)))
    return frameDirs


def checkDir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def argument(ims, zoom, flip, rotate):
    imsArg = []
    for im in ims:
        s = im.size
        # print(im.size)
        im = im.resize([int(d * zoom) for d in s], Image.BICUBIC)
        # print(im.size)
        if flip:
            im = im.transpose(Image.FLIP_TOP_BOTTOM)
        # print(im.size)
        if rotate == 1:
            im = im.transpose(Image.ROTATE_90)
        if rotate == 2:
            im = im.transpose(Image.ROTATE_180)
        if rotate == 3:
            im = im.transpose(Image.ROTATE_270)
        # print(im.size)
        imsArg.append(im)
    # for im in imsArg:
    #     print(im.size)
    return imsArg


def randomCrop(ims, cropWidth):
    imsCrop = []
    w, h = ims[0].size
    wRange = w - cropWidth
    hRange = h - cropWidth
    wStart = random.randint(0, wRange)
    hStart = random.randint(0, hRange)
    # print(wStart, hStart)
    for im in ims:
        im = im.crop((wStart, hStart, wStart + cropWidth, hStart + cropWidth))
        imsCrop.append(im)
    return imsCrop

from matplotlib import pyplot as plt
def saveBatch(batch, dir):
    print('Saving to ' + dir)
    writer = tf.python_io.TFRecordWriter(dir)
    for patch in batch:
        feature = {}
        for i, im in enumerate(patch):
            if i == 0:
                imName = 'gt'
            else:
                imName = 'input_%d' % (i - 1)
            pngStr = BytesIO()
            im.save(pngStr, format='png')
            pngStr = pngStr.getvalue()
            feature[imName] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[pngStr]))
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
    writer.close()
    pass


def generateBatches(frameDirs, saveFolder, nCrop):
    # batchesDone = glob.glob(os.path.join(saveFolder, '*.tfrecords'))
    # batchesDone = [os.path.split(batcheDone)[1] for batcheDone in batchesDone]
    # batchesDone.sort()
    # if len(batchesDone) > 0:
    #     iBatchStart = int(re.findall('\d{5}', batchesDone[-1])[0])
    # else:
    #     iBatchStart = 0

    # start generating
    batch = []
    iBatch = 0
    nBatches = nCrop * len(frameDirs) * nArguments / batchSize
    tic = time.time()
    for frameDir in frameDirs:
        inputDirs = frameDir[0]
        GTDir = frameDir[1]
        gtIm = Image.open(GTDir)
        inputIms = [Image.open(inputDir) for inputDir in inputDirs]
        for zoom in argumentZoom:
            for flip in range(2):
                for rotate in range(4):
                    gtImArg = argument([gtIm], zoom, flip, rotate)
                    inputImsArg = argument(inputIms, zoom, flip, rotate)
                    for iCrop in range(nCrop):
                        patchCroped = randomCrop(gtImArg + inputImsArg, cropWidth)
                        batch.append(patchCroped)
                        if len(batch) >= batchSize:
                            saveBatch(batch, os.path.join(
                                saveFolder,
                                'batch_width%d_size%d_%05d.tfrecords' % (cropWidth, batchSize, iBatch)
                            ))
                            iBatch += 1
                            batch = []
                            ms = (time.time() - tic) * (nBatches - iBatch) / 60
                            tic = time.time()
                            hours = math.floor(ms / 60)
                            mins = ms % 60
                            print(
                                '%.2f%% %d/%d, %d hours %.1f minutes left.' %
                                ((iBatch) / nBatches * 100, iBatch, nBatches, hours, mins)
                            )

    # for iBatch in range(iBatchStart, int(nBatches)):
    #     batch = []
    #     for iFrame in [random.randint(0, len(frameDirs) - 1) for i in range(batchSize)]:
    #         zoom = random.choice(argumentZoom)
    #         flip = random.randint(0, 1)
    #         rotate = random.randint(0, 3)
    #
    #         inputDirs = frameDirs[iFrame][0]
    #         GTDir = frameDirs[iFrame][1]
    #         gtIm = Image.open(GTDir)
    #         # print(GTDir)
    #         gtIm = argument([gtIm], zoom, flip, rotate)
    #         inputIms = [Image.open(inputDir) for inputDir in inputDirs]
    #         # print(inputDirs)
    #         inputIms = argument(inputIms, zoom, flip, rotate)
    #         patchCroped = randomCrop(gtIm + inputIms, cropWidth)
    #         # print(len(patchCroped))
    #         batch.append(patchCroped)
    #     saveBatch(batch, os.path.join(
    #         saveFolder,
    #         'batch_width%d_size%d_%05d.tfrecords' % (cropWidth, batchSize, iBatch)
    #     ))
    #     ms = (time.time() - tic) * (nBatches - iBatch) / 60
    #     tic = time.time()
    #     hours = math.floor(ms / 60)
    #     mins = ms % 60
    #     print(
    #         '%.2f%% %d/%d, %d hours %.1f minutes left.' %
    #         ((iBatch + 1) / nBatches * 100, iBatch + 1, nBatches, hours, mins)
    #     )


for iAlignment in range(len(alignments)):
    alignment = alignments[iAlignment]

    # scan frames
    trainset = list(set(videoNames[iAlignment]).difference(set(validset)))
    trainset.sort()
    frameDirs = scanFrames(inputFolders[iAlignment], gtDir, trainset)

    # generate batches
    saveAlignFolder = os.path.join(saveDir, saveFolderPrefixTrain + alignment)
    checkDir(saveAlignFolder)
    generateBatches(frameDirs, saveAlignFolder, nCrop)

    # scan validset frames
    validset.sort()
    frameDirs = scanFrames(inputFolders[iAlignment], gtDir, validset)

    # generate validset batches
    saveAlignFolder = os.path.join(saveDir, saveFolderPrefixValid + alignment)
    checkDir(saveAlignFolder)
    generateBatches(frameDirs, saveAlignFolder, nCrop)
