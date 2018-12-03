import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=251, help='rand seed')

parser.add_argument('--model', type=str, default='', help='model name')
parser.add_argument('--model_param_load', type=str, default='', help='weight file to load')
parser.add_argument('--bn_meanstd_load', type=str, default='', help='batch normalization file to load')
parser.add_argument('--optimstate_load', type=str, default='', help='state of optim.adam to load')

parser.add_argument('--num_frames', type=int, default=5, help='number of frames in input stack')
parser.add_argument('--num_channels', type=int, default=3, help='rgb input')
parser.add_argument('--max_intensity', type=int, default=255, help='maximum intensity of input')

# parser.add_argument('--data_root', type=str, default='', help='folder for datasets')
parser.add_argument('--data_trainset', type=str, default='', help='folder for transet data')
parser.add_argument('--data_validset', type=str, default='', help='folder for validset data')
parser.add_argument('--trainset_size', type=int, default=61,
                    help='size of trainset (if is larger than dataset, use all dataset as trainset)')
parser.add_argument('--batch_size', type=int, default=64,
                    help='size of batch sampled from batch files (if is larger than size from batch files, use entire batch every iteration)')
parser.add_argument('--it_max', type=int, default=80000, help='max number of iterations')

parser.add_argument('--model_param', type=str, default='', help='weight file')
parser.add_argument('--bn_meanstd', type=str, default='', help='batch normalization file')
parser.add_argument('--optimstate', type=str, default='', help='state of optim.adam')
parser.add_argument('--save_every', type=int, default=500, help='auto save every save_every iterations')
parser.add_argument('--log', type=str, default='', help='dir to save log')
parser.add_argument('--log_every', type=int, default=1, help='log every log_every iterations')

parser.add_argument('--reset_lr', type=int, default=None, help='reset lr to reset_lr instead of default or saved lr')
parser.add_argument('--reset_state', type=int, default=0, help='reset optim state')
parser.add_argument('--decay_from', type=int, default=24000, help='decay learning rate from decay_from iterations')
parser.add_argument('--decay_every', type=int, default=8000, help='decay learning rate every decay_every iteration')
parser.add_argument('--decay_rate', type=int, default=0.5, help='decay learning rate')

parser.add_argument('--overfit_batches', type=int, default=0, help='overfit batch num for debuging')
parser.add_argument('--overfit_patches', type=int, default=0, help='overfit patch num, valid if overfit_batches = 1')
parser.add_argument('--overfit_out', type=str, default='', help='overfit output on trainset')

args = parser.parse_args()

# scan batches
import os
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

trainsetDirs = []
validsetDirs = []
if args.data_trainset != '':
    trainsetDirs = scanSetFolder(args.data_trainset)
    showBatchDirExamples(trainsetDirs)
if args.data_validset != '':
    validsetDirs = scanSetFolder(args.data_validset)
    showBatchDirExamples(validsetDirs)


# preload trainset if it's small


# load model
import tensorflow as tf
import importlib
modelDir = args.model
print( "Loading model %s" % modelDir )
spec = importlib.util.spec_from_file_location('model', modelDir)
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)
tf.reset_default_graph()
input = tf.placeholder(tf.float32, shape=(None, 128, 128, 15))
training = tf.placeholder(tf.bool, shape=(), name='training')
output = model.create_model(input, training)
print("Model loaded")


# params setting
max_intensity = args.max_intensity
batchSize = args.batch_size
decayFrom = args.decay_from
decayEvery = args.decay_every
decayRate = args.decay_rate
lrMin = 1e-6
itMax = args.it_max

print("Params setting: max_intensity = %f" % max_intensity)
print("                    batchSize = %d" % batchSize)
print("                    decayFrom = %d" % decayFrom)
print("                   decayEvery = %d" % decayEvery)
print("                    decayRate = %f" % decayRate)
print("                        itMax = %d" % itMax)
print("                        lrMin = %f" % lrMin)

pass


# load params
if args.reset_lr != None:
    learningRate = args.reset_lr
else:
    learningRate = 0.005
print('Learning rate = %f' % learningRate)


# load random batch sample
from random import choice
import scipy.io as sio
import numpy as np
def loadRandomBatchFrom(batchDirs, batchSize=None):
    sampleDir = choice(batchDirs)
    batchSample = sio.loadmat(sampleDir)
    batchInputRaw = batchSample['batchInputTorch']
    batchGTRaw = batchSample['batchGTTorch']

    if batchSize is None:
        batchSize = batchInputRaw.shape[0]
    if batchSize == batchInputRaw.shape[0]:
        batchInput = batchInputRaw
        batchGT = batchGTRaw
    else:
        idxs = np.random.choice(batchInputRaw.shape[0], batchSize, replace=False)
        print(idxs)
        batchInput = batchInputRaw[idxs, :, :, :]
        batchGT = batchGTRaw[idxs, :, :, :]
    # change to NHWC
    batchInput = batchInput.transpose((0, 2, 3, 1)).astype(np.float32) / max_intensity
    batchGT = batchGT.transpose((0, 2, 3, 1)).astype(np.float32) / max_intensity

    return batchInput, batchGT


# save params
pass


# preload batch if overfit_batches = 1


# prepare
gt = tf.placeholder(tf.float32, shape=(None, 128, 128, 3))
loss = tf.losses.mean_squared_error(
    gt,
    output
)
optimizer = tf.train.AdamOptimizer(
    learning_rate=learningRate
)
updateOps = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(updateOps):
    trainOp = optimizer.minimize(loss)

import matplotlib.pyplot as plt
def showBatchIm(batch,i):
    im = batch[i, :, :, :]
    plt.imshow(im)
    plt.show()


# train
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
if args.log != '':
    writer = tf.summary.FileWriter(args.log, sess.graph)
    tf.summary.scalar('loss', loss)
    merged = tf.summary.merge_all()
    print('Logging to: '+args.log)
sess.run(tf.global_variables_initializer())
for it in range(1000):
    batchInput, batchGT = loadRandomBatchFrom(trainsetDirs)
    batchLoss, _ = sess.run(
        (loss, trainOp),
        feed_dict={
            training: True,
            input: batchInput,
            gt: batchGT
        }
    )
    print('it %d, loss %f' % (it, batchLoss))
    if args.log != '' and it % 5 == 0:
        rs = sess.run(merged, feed_dict={
            training: True,
            input: batchInput,
            gt: batchGT
        })
        writer.add_summary(rs, it)
        print('Logging to: ' + args.log)

writer.close()


# test net

batchInput, batchGT = loadRandomBatchFrom(trainsetDirs,1)
inputIm = batchInput[:, :, :, 6:9]
showBatchIm(inputIm,0)
showBatchIm(batchGT,0)

def testNet(batchInput,sess):
    batchOutput, batchLoss = sess.run(
        (output, loss),
        feed_dict={
            training: False,
            input: batchInput,
            gt: batchGT
        }
    )
    print('loss = %f' % batchLoss)
    showBatchIm(batchOutput,0)
testNet(batchInput, sess)

# test on trainset for overfitting
