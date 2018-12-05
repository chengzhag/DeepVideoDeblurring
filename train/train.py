import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=251, help='rand seed')

parser.add_argument('--model', type=str, default='', help='model name')
parser.add_argument('--ckp_dir_load', type=str, default='', help='checkpoint dir to load')

# parser.add_argument('--num_frames', type=int, default=5, help='number of frames in input stack')
# parser.add_argument('--num_channels', type=int, default=3, help='rgb input')
parser.add_argument('--max_intensity', type=int, default=255, help='maximum intensity of input')

# parser.add_argument('--data_root', type=str, default='', help='folder for datasets')
parser.add_argument('--data_trainset', type=str, default='', help='folder for transet data')
parser.add_argument('--data_validset', type=str, default='', help='folder for validset data')
parser.add_argument('--batch_size', type=int, default=64,
                    help='size of batch sampled from batch files (if is larger than size from batch files, use entire batch every iteration)')
parser.add_argument('--it_max', type=int, default=80000, help='max number of iterations')

parser.add_argument('--ckp_dir', type=str, default='', help='checkpoint dir to save')
parser.add_argument('--save_every', type=int, default=500, help='auto save every save_every iterations')
parser.add_argument('--log', type=str, default='', help='dir to save log')
parser.add_argument('--log_every', type=int, default=1, help='log every log_every iterations')

parser.add_argument('--reset_lr', type=float, default=None, help='reset lr to reset_lr instead of default or saved lr')
# parser.add_argument('--reset_state', type=int, default=0, help='reset optim state')
parser.add_argument('--decay_from', type=int, default=24000, help='decay learning rate from decay_from iterations')
parser.add_argument('--decay_every', type=int, default=8000, help='decay learning rate every decay_every iteration')
parser.add_argument('--decay_rate', type=int, default=0.5, help='decay learning rate')

# parser.add_argument('--overfit_batches', type=int, default=0, help='overfit batch num for debuging')
# parser.add_argument('--overfit_patches', type=int, default=0, help='overfit patch num, valid if overfit_batches = 1')
# parser.add_argument('--overfit_out', type=str, default='', help='overfit output on trainset')

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
print("Loading model %s" % modelDir)
specImport = importlib.util.spec_from_file_location('model', modelDir)
model = importlib.util.module_from_spec(specImport)
specImport.loader.exec_module(model)

tf.reset_default_graph()
inputPh = tf.placeholder(tf.float32, shape=(None, 128, 128, 15), name='input')
gtPh = tf.placeholder(tf.float32, shape=(None, 128, 128, 3), name='gt')
trainingPh = tf.placeholder(tf.bool, shape=(), name='training')
learningRatePh = tf.placeholder(tf.float32, shape=(), name='learningRate')
learningRateT = tf.Variable(0.005,trainable=False,dtype=tf.float32)
global_step = tf.train.create_global_step()
optimizer = tf.train.AdamOptimizer(learning_rate=learningRatePh)
outputT = model.create_model(inputPh, trainingPh)
lossT = tf.losses.mean_squared_error(gtPh, outputT)

updateOps = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(updateOps):
    trainOp = optimizer.minimize(lossT, global_step=global_step)
print("Model loaded")


# params setting
maxIntensity = args.max_intensity
batchSize = args.batch_size
decayFrom = args.decay_from
decayEvery = args.decay_every
decayRate = args.decay_rate
lrMin = 1e-6
itMax = args.it_max

print("Params setting: max_intensity = %f" % maxIntensity)
print("                    batchSize = %d" % batchSize)
print("                    decayFrom = %d" % decayFrom)
print("                   decayEvery = %d" % decayEvery)
print("                    decayRate = %f" % decayRate)
print("                        itMax = %d" % itMax)
print("                        lrMin = %f" % lrMin)

# load params
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
saver = tf.train.Saver()
if args.ckp_dir_load != '':
    loadDir = os.path.join(args.ckp_dir_load, 'checkpoints')
    loadDir = tf.train.latest_checkpoint(args.ckp_dir_load)
    try:
        print('Loading params from: ' + loadDir)
        saver.restore(sess, loadDir)
    except:
        print('Load params failed!')
        sess.run(tf.global_variables_initializer())
else:
    sess.run(tf.global_variables_initializer())

if args.reset_lr != None:
    learningRateT.load(args.reset_lr, sess)

print('Learning rate = %f' % sess.run(learningRateT))


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
    batchInput = batchInput.transpose((0, 2, 3, 1)).astype(np.float32) / maxIntensity
    batchGT = batchGT.transpose((0, 2, 3, 1)).astype(np.float32) / maxIntensity

    return batchInput, batchGT


# preload batch if overfit_batches = 1



# train
import time
if args.log != '':
    lossAvg = 0
    writer = tf.summary.FileWriter(args.log, sess.graph)
    lossAvgPh = tf.placeholder(dtype=tf.float32)
    tf.summary.scalar('loss', lossAvgPh)
    tf.summary.scalar('learning rate', learningRateT)
    tf.summary.image('inputIm',tf.slice(inputPh, [0, 0, 0, 6], [-1, -1, -1, 3]))
    tf.summary.image('outputIm', outputT)
    merged = tf.summary.merge_all()
    print('Logging to: ' + args.log)

tic = time.time()
it = tf.train.global_step(sess, global_step)
avgSpeed = None
while 1:
    if it >= itMax:
        break
    if it >= decayFrom and (it - decayFrom) % decayEvery == 0:
        sess.run(learningRateT.assign(learningRateT * decayRate))
    if learningRateT.eval(sess) < lrMin:
        learningRateT.load(lrMin, sess)
    batchInput, batchGT = loadRandomBatchFrom(trainsetDirs)
    batchLoss, _ = sess.run(
        (lossT, trainOp),
        feed_dict={
            learningRatePh: learningRateT.eval(sess),
            trainingPh: True,
            inputPh: batchInput,
            gtPh: batchGT
        }
    )
    it = tf.train.global_step(sess, global_step)

    toc = time.time()
    speed = 1 / (toc - tic)
    avgSpeed = avgSpeed or speed
    avgSpeed = 0.1 * speed + 0.9 * avgSpeed
    timeLeft = (itMax - it) / avgSpeed
    toc = tic
    tic = time.time()
    timeLeftMin = timeLeft / 60
    print('it: %d, loss: %f, lr: %f, spd: %.2f it/s, avgSpd: %.2f it/s, left: %d h %.1f min' %
          (it, batchLoss, learningRateT.eval(sess), speed, avgSpeed, timeLeftMin / 60, timeLeftMin % 60))
    lossAvg = lossAvg + batchLoss
    if args.log != '' and it % args.log_every == 0:
        lossAvg = lossAvg/args.log_every
        rs = sess.run(
            merged,
            feed_dict={
                learningRatePh: learningRateT.eval(sess),
                trainingPh: True,
                inputPh: batchInput,
                gtPh: batchGT,
                lossAvgPh:lossAvg
            })
        writer.add_summary(rs, it)
        print('Logging to: ' + args.log)
        lossAvg = 0

    if it % args.save_every == 0 or it == itMax:
        if args.ckp_dir != '':
            if not os.path.exists(args.ckp_dir):
                os.makedirs(args.ckp_dir)
            savePath = saver.save(
                sess,
                os.path.join(args.ckp_dir, 'checkpoints'),
                global_step=it
            )
            print("Model saved in path: %s" % savePath)
        else:
            print('Use param --ckp_dir to specify path to save!')

writer.close()


# test net
import matplotlib.pyplot as plt

def showBatchIm(batch, i):
    im = batch[i, :, :, :]
    plt.imshow(im)
    plt.show()

batchInput, batchGT = loadRandomBatchFrom(trainsetDirs, 1)
inputIm = batchInput[:, :, :, 6:9]
showBatchIm(inputIm, 0)
showBatchIm(batchGT, 0)

def testNet(batchInput, batchGT, sess):
    batchOutput, batchLoss = sess.run(
        (outputT, lossT),
        feed_dict={
            trainingPh: False,
            inputPh: batchInput,
            gtPh: batchGT
        }
    )
    print('loss = %f' % batchLoss)
    showBatchIm(batchOutput, 0)


testNet(batchInput, batchGT, sess)

# test on trainset for overfitting
