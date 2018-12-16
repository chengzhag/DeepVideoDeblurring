import argparse
import os
import tensorflow as tf
import math
import scipy.misc
import numpy as np
import importlib
import models.model as model
import glob
import random

def parseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default=None, help='model name')
    parser.add_argument('--ckp_dir', type=str, default=None,
                        help='checkpoint dir to save or load for validating and testing')
    parser.add_argument('--ckp_dir_load', type=str, default=None, help='checkpoint dir to load for training')

    parser.add_argument('--data_trainset', type=str, default=None, help='folder for transet data')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='size of batch sampled from batch files (if is larger than size from batch files, use entire batch every iteration)')
    parser.add_argument('--it_max', type=int, default=80000, help='max number of iterations')
    parser.add_argument('--reset_lr', type=float, default=None,
                        help='reset lr to reset_lr instead of default or saved lr')
    parser.add_argument('--decay_from', type=int, default=24000, help='decay learning rate from decay_from iterations')
    parser.add_argument('--decay_every', type=int, default=8000, help='decay learning rate every decay_every iteration')
    parser.add_argument('--decay_rate', type=int, default=0.5, help='decay learning rate')
    parser.add_argument('--save_every', type=int, default=100, help='auto save every save_every iterations')
    parser.add_argument('--log_every', type=int, default=100, help='log every log_every iterations')

    parser.add_argument('--data_testset', type=str, default=None, help='folder for testset data')
    parser.add_argument('--output_dir', type=str, default=None, help='folder to output test results')

    parser.add_argument('--data_validset', type=str, default=None, help='folder for validset data')
    parser.add_argument('--num_valid', type=int, default=100, help='num of batches used for testing loss')

    parser.add_argument('--num_gpus', type=int, default=1, help='num of gpus')

    return parser.parse_args()


def scanSetFolder(folder):
    dirs = glob.glob(os.path.join(folder,'*.tfrecords'))
    dirs.sort()
    names = [os.path.split(dir)[1] for dir in dirs]
    return dirs, names


def showDirExamples(dirs):
    num = min(10, len(dirs))
    print("First %d dir examples of %d dirs:" % (num, len(dirs)))
    for i in range(num):
        print('\t' + dirs[i])

def loadModel(args):
    modelDir = args.model
    print("Loading model %s" % modelDir)
    specImport = importlib.util.spec_from_file_location('createFcn', modelDir)
    createFcn = importlib.util.module_from_spec(specImport)
    specImport.loader.exec_module(createFcn)
    deblur = model.Deblur(createFcn.create_model,nGPUs=args.num_gpus)
    print("Model loaded")

    if args.ckp_dir is not None:
        print("Loading params from %s" % args.ckp_dir)
        try:
            deblur.load(args.ckp_dir)
        except:
            print('No checkpoint or incompatible checkpoint in ckp_dir!')
        else:
            return deblur

    if args.ckp_dir_load is not None:
        try:
            deblur.load(args.ckp_dir_load)
        except:
            print('No checkpoint or incompatible checkpoint in ckp_dir_load!')
        else:
            return deblur

    deblur.initialize()
    return deblur

def train(args):
    # scan dataset
    print('Scanning trainset files from %s' % args.data_trainset)
    trainsetDirs, _ = scanSetFolder(args.data_trainset)
    print('Found %d trainset files' % len(trainsetDirs))
    random.shuffle(trainsetDirs)
    showDirExamples(trainsetDirs)

    # load model
    deblur = loadModel(args)

    # train
    deblur.train(
        trainsetDirs,
        saveEvery=args.save_every,
        ckpDir=args.ckp_dir,
        logEvery=args.log_every,
        batchSize=args.batch_size,
        decayFrom=args.decay_from,
        decayEvery=args.decay_every,
        decayRate=args.decay_rate,
        lrMin=1e-6,
        itMax=args.it_max,
        lr=args.reset_lr
    )

def valid(args):
    print('Scanning validset files from %s' % args.data_validset)
    validsetDirs, _ = scanSetFolder(args.data_validset)
    print('Found %d validset files' % len(validsetDirs))
    random.shuffle(validsetDirs)
    showDirExamples(validsetDirs)

    # load model
    deblur = loadModel(args)

    lossAvg = deblur.valid(validsetDirs, args.num_valid, args.batch_size)
    print('Loss on dataset is %f' % lossAvg)

def test(args):
    # scan dataset
    print('Scanning testset videos from %s' % args.data_trainset)
    videoDirs, videoNames = scanSetFolder(args.data_testset)
    print('Found %d testset videos' % len(videoDirs))
    showDirExamples(videoDirs)

    # load model
    deblur = loadModel(args)

    nAdjFrame = 5
    maxIntensity = 255
    for videoDir,videoName in zip(videoDirs,videoNames):
        outputFolder = os.path.join(args.output_dir, videoName)
        if not os.path.exists(outputFolder):
            os.makedirs(outputFolder)
        print('Scanning frames from ' + videoDir)
        frameNames = os.listdir(os.path.join(videoDir,'image_0'))
        for iFrame, frameName in enumerate(frameNames):
            adjFrames = []
            for iAdjFrame in range(nAdjFrame):
                iAdjFrameFolder = 'image_%d' % (iAdjFrame-math.floor(nAdjFrame/2))
                frameDir = os.path.join(videoDir, iAdjFrameFolder, frameName)
                frameInput = scipy.misc.imread(frameDir)
                adjFrames.append(frameInput)
            input = np.concatenate(adjFrames, axis=-1)
            inputH,inputW,inputC = input.shape
            inputPadH = 8 * math.ceil(inputH / 8)
            inputPadW = 8 * math.ceil(inputW / 8)
            inputPad = np.pad(
                input,
                ((0,inputPadH-inputH),(0,inputPadW-inputW),(0,0)),
                'edge'
            )
            inputPad = inputPad.astype(np.float32)/maxIntensity
            inputPad = np.expand_dims(inputPad, 0)
            # print(input.shape)
            # print(inputPad.shape)
            outputPad = deblur.predictIm(inputPad)
            # print(outputPad.shape)
            output = outputPad[0,:inputH,:inputW,:]
            # print(output.shape)
            # deblur.showBatchIm(outputPad)
            scipy.misc.imsave(os.path.join(outputFolder, frameName), output)
            print('%d / %d' % (iFrame, len(frameNames)))



def main(_):
    args = parseArgs()

    if args.data_trainset is not None:
        train(args)
    elif args.data_validset is not None:
        valid(args)
    elif args.data_testset is not None:
        test(args)


if __name__ == '__main__':
    tf.app.run()
