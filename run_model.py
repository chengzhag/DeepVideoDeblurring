import argparse
import os
import tensorflow as tf


def parseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default=None, help='model name')
    parser.add_argument('--ckp_dir_load', type=str, default=None, help='checkpoint dir to load')

    parser.add_argument('--data_trainset', type=str, default=None, help='folder for transet data')
    parser.add_argument('--data_validset', type=str, default=None, help='folder for validset data')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='size of batch sampled from batch files (if is larger than size from batch files, use entire batch every iteration)')
    parser.add_argument('--it_max', type=int, default=80000, help='max number of iterations')

    parser.add_argument('--ckp_dir', type=str, default=None,
                        help='checkpoint dir to save, if not empty, scan for checkpoint')
    parser.add_argument('--save_every', type=int, default=300, help='auto save every save_every iterations')
    parser.add_argument('--log_every', type=int, default=20, help='log every log_every iterations')

    parser.add_argument('--reset_lr', type=float, default=None,
                        help='reset lr to reset_lr instead of default or saved lr')
    parser.add_argument('--decay_from', type=int, default=24000, help='decay learning rate from decay_from iterations')
    parser.add_argument('--decay_every', type=int, default=8000, help='decay learning rate every decay_every iteration')
    parser.add_argument('--decay_rate', type=int, default=0.5, help='decay learning rate')

    return parser.parse_args()


def scanSetFolder(folder):
    print('Scanning trainset batches from %s' % folder)
    dirs = os.listdir(folder)
    dirs = [os.path.join(folder, dir) for dir in dirs]
    print('Found %d trainset batches' % len(dirs))
    return dirs


def showBatchDirExamples(dirs):
    num = min(10, len(dirs))
    print("First %d batches example of %d trainset Batch:" % (num, len(dirs)))
    for i in range(num):
        print('\t' + dirs[i])


def main(_):
    args = parseArgs()

    trainsetDirs = []
    if args.data_trainset is not None:
        trainsetDirs = scanSetFolder(args.data_trainset)
        showBatchDirExamples(trainsetDirs)

    validsetDirs = []
    if args.data_validset is not None:
        validsetDirs = scanSetFolder(args.data_validset)
        showBatchDirExamples(validsetDirs)

    # load model
    import importlib
    import models.model as model

    modelDir = args.model
    print("Loading model %s" % modelDir)
    specImport = importlib.util.spec_from_file_location('createFcn', modelDir)
    createFcn = importlib.util.module_from_spec(specImport)
    specImport.loader.exec_module(createFcn)
    deblur = model.Deblur(createFcn.create_model)
    print("Model loaded")

    if args.ckp_dir is not None:
        try:
            deblur.load(args.ckp_dir)
        except:
            print('No checkpoint or incompatible checkpoint in ckp_dir!')

    if args.ckp_dir_load is not None:
        try:
            deblur.load(args.ckp_dir_load)
        except:
            print('No checkpoint or incompatible checkpoint in ckp_dir_load!')

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

    # test
    # deblur.showRandomBatchTest(trainsetDirs)


if __name__ == '__main__':
    tf.app.run()
