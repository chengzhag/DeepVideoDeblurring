import tensorflow as tf
import os
from random import choice
import scipy.io as sio
import numpy as np
import time
import matplotlib.pyplot as plt

class Deblur(object):
    def __init__(self, createFcn, maxIntensity=255, nGPUs=1):
        self.maxIntensity = maxIntensity

        tf.reset_default_graph()

        self.inputPh = tf.placeholder(tf.float32, shape=(None, None, None, 15), name='input')
        self.gtPh = tf.placeholder(tf.float32, shape=(None, None, None, 3), name='gt')
        self.trainingPh = tf.placeholder(tf.bool, shape=(), name='training')
        self.learningRateV = tf.Variable(0.005, trainable=False, dtype=tf.float32, name='learning_rate')
        self.globalStepV = tf.train.create_global_step()
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learningRateV, name='optim')

        print('Using %d GPUs' % nGPUs)
        if nGPUs > 1:
            inputSplitsT = tf.split(self.inputPh, nGPUs)
            gtSplitsT = tf.split(self.gtPh, nGPUs)
            towerGradsT = []
            towerLossT = []
            towerOutputT = []

            for iGPU in range(nGPUs):
                with tf.variable_scope(tf.get_variable_scope(), reuse=iGPU > 0):
                    with tf.device("/gpu:%d" % iGPU):
                        with tf.name_scope("tower_%d" % iGPU):
                            outputT = createFcn(inputSplitsT[iGPU], self.trainingPh)
                            lossT = tf.losses.mean_squared_error(gtSplitsT[iGPU], outputT)
                            gradsT = self.optimizer.compute_gradients(lossT)
                            towerGradsT.append(gradsT)
                            towerLossT.append(lossT)
                            towerOutputT.append(outputT)

            def average_gradients(tower_grads):
                average_grads = []
                for grad_and_vars in zip(*tower_grads):
                    grads = []
                    for g, _ in grad_and_vars:
                        expend_g = tf.expand_dims(g, 0)
                        grads.append(expend_g)
                    grad = tf.concat(grads, 0)
                    grad = tf.reduce_mean(grad, 0)
                    v = grad_and_vars[0][1]
                    grad_and_var = (grad, v)
                    average_grads.append(grad_and_var)
                return average_grads

            gradsT = average_gradients(towerGradsT)
            self.lossT = tf.reduce_mean(towerLossT)
            self.outputT = tf.concat(towerOutputT, 0)

            updateOps = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(updateOps):
                self.trainOp = self.optimizer.apply_gradients(
                    gradsT,
                    global_step=self.globalStepV
                )
        else:
            with tf.device("/gpu:0"):
                self.outputT = createFcn(self.inputPh, self.trainingPh)
                self.lossT = tf.losses.mean_squared_error(self.gtPh, self.outputT)

                updateOps = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(updateOps):
                    self.trainOp = self.optimizer.minimize(
                        self.lossT,
                        global_step=self.globalStepV
                    )

        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))

    def initialize(self):
        self.sess.run(tf.global_variables_initializer())

    def load(self, ckpDir):
        loadDir = tf.train.latest_checkpoint(ckpDir)
        print('Loading params from: ' + loadDir)
        saver = tf.train.Saver()
        saver.restore(self.sess, loadDir)

    def valid(self, validsetDirs, nValid, batchSize):
        batchT = self.loadDataset(validsetDirs, batchSize)

        tic = time.time()
        avgSpeed = None
        lossAvg = 0
        for iBatch in range(nValid):
            batchInput, batchGT = self.sess.run(batchT)
            batchLoss = self.sess.run(
                self.lossT,
                feed_dict={
                    self.trainingPh: False,
                    self.inputPh: batchInput,
                    self.gtPh: batchGT
                }
            )
            lossAvg += batchLoss
            toc = time.time()
            speed = 1 / (toc - tic)
            avgSpeed = avgSpeed or speed
            avgSpeed = 0.1 * speed + 0.9 * avgSpeed
            timeLeft = (nValid - iBatch) / avgSpeed
            toc = tic
            tic = time.time()
            timeLeftMin = timeLeft / 60
            print('iBatch: %d, lossAvg: %f, spd: %.2f it/s, avgSpd: %.2f it/s, left: %d h %.1f min' %
                  (iBatch, lossAvg/(iBatch+1), speed, avgSpeed, timeLeftMin / 60, timeLeftMin % 60))
        return lossAvg/nValid

    def train(
            self,
            trainsetDirs,
            ckpDir=None,
            saveEvery=1000,
            logEvery=20,
            batchSize=64,
            decayFrom=24000,
            decayEvery=8000,
            decayRate=0.5,
            lrMin=1e-6,
            itMax=80000,
            lr=None,
    ):
        print("                    batchSize = %d" % batchSize)
        print("                    decayFrom = %d" % decayFrom)
        print("                   decayEvery = %d" % decayEvery)
        print("                    decayRate = %f" % decayRate)
        print("                        itMax = %d" % itMax)
        print("                        lrMin = %f" % lrMin)

        if lr is not None:
            self.learningRateV.load(lr, self.sess)
            print('Learning rate reseted to %f' % self.sess.run(self.learningRateV))

        if ckpDir is not None:
            lossAvg = 0
            writer = tf.summary.FileWriter(os.path.join(ckpDir, 'logs'), self.sess.graph)
            lossAvgPh = tf.placeholder(dtype=tf.float32)
            tf.summary.scalar('loss', lossAvgPh)
            tf.summary.scalar('learningRate', self.learningRateV)
            tf.summary.image('inputIm', tf.slice(self.inputPh, [0, 0, 0, 6], [-1, -1, -1, 3]))
            tf.summary.image('outputIm', self.outputT)
            merged = tf.summary.merge_all()
            print('Logging to: ' + ckpDir)

        batchT = self.loadDataset(trainsetDirs, batchSize)

        saver = tf.train.Saver()
        if ckpDir is not None:
            if not os.path.exists(ckpDir):
                os.makedirs(ckpDir)

        tic = time.time()
        it = tf.train.global_step(self.sess, self.globalStepV)
        avgSpeed = None
        while 1:
            if it >= itMax:
                break
            if it >= decayFrom and (it - decayFrom) % decayEvery == 0:
                self.sess.run(self.learningRateV.assign(self.learningRateV * decayRate))
            if self.learningRateV.eval(self.sess) < lrMin:
                self.learningRateV.load(lrMin, self.sess)
            batchInput, batchGT = self.sess.run(batchT)
            batchLoss, _ = self.sess.run(
                (self.lossT, self.trainOp),
                feed_dict={
                    self.trainingPh: True,
                    self.inputPh: batchInput,
                    self.gtPh: batchGT
                }
            )
            it = tf.train.global_step(self.sess, self.globalStepV)

            toc = time.time()
            speed = 1 / (toc - tic)
            avgSpeed = avgSpeed or speed
            avgSpeed = 0.1 * speed + 0.9 * avgSpeed
            timeLeft = (itMax - it) / avgSpeed
            toc = tic
            tic = time.time()
            timeLeftMin = timeLeft / 60
            print('it: %d, loss: %f, lr: %f, spd: %.2f it/s, avgSpd: %.2f it/s, left: %d h %.1f min' %
                  (it, batchLoss, self.learningRateV.eval(self.sess), speed, avgSpeed, timeLeftMin / 60,
                   timeLeftMin % 60))
            if ckpDir is not None: lossAvg = lossAvg + batchLoss
            if ckpDir is not None and it % logEvery == 0:
                lossAvg = lossAvg / logEvery
                rs = self.sess.run(
                    merged,
                    feed_dict={
                        self.trainingPh: False,
                        self.inputPh: batchInput,
                        self.gtPh: batchGT,
                        lossAvgPh: lossAvg
                    })
                print('Logging to: ' + ckpDir)
                writer.add_summary(rs, it)
                lossAvg = 0

            if it % saveEvery == 0 or it == itMax:
                if ckpDir is not None:
                    savePath = saver.save(
                        self.sess,
                        os.path.join(ckpDir, 'checkpoints'),
                        global_step=it
                    )
                    print("Model saved in path: %s" % savePath)
                else:
                    print('Use param --ckp_dir to specify path to save!')

    def showBatchIm(self, batch, i=0):
        im = batch[i, :, :, :]
        plt.imshow(im)
        plt.show()

    def showRandomBatchTest(self, batchDirs):
        batchInput, batchGT = self.loadRandomBatchFrom(batchDirs)
        inputIm = batchInput[:, :, :, 6:9]
        self.showBatchIm(inputIm)
        self.showBatchIm(batchGT)

        batchOutput, batchLoss = self.testBatch(batchInput, batchGT)
        print('loss = %f' % batchLoss)
        self.showBatchIm(batchOutput)

    def predictIm(self, im, patchbypatch=False):
        if not patchbypatch:
            return self.predictBatch(im)
        else:
            pass

    def predictBatch(self, batchInput):
        batchOutput = self.sess.run(
            self.outputT,
            feed_dict={
                self.trainingPh: False,
                self.inputPh: batchInput,
            }
        )
        return batchOutput

    def testBatch(self, batchInput, batchGT):
        batchOutput, batchLoss = self.sess.run(
            (self.outputT, self.lossT),
            feed_dict={
                self.trainingPh: False,
                self.inputPh: batchInput,
                self.gtPh: batchGT
            }
        )
        return batchOutput, batchLoss

    def loadRandomBatchFrom(self, batchDirs, batchSize=None):
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
        batchInput = batchInput.transpose((0, 2, 3, 1)).astype(np.float32) / self.maxIntensity
        batchGT = batchGT.transpose((0, 2, 3, 1)).astype(np.float32) / self.maxIntensity

        return batchInput, batchGT

    def loadDataset(self, batchDirs, batchSize):
        with tf.name_scope('load'):
            datasetRaw = tf.data.TFRecordDataset(batchDirs)

            def parse_function(example_proto):
                feature = {}
                for i in range(6):
                    if i == 0:
                        imName = 'gt'
                    else:
                        imName = 'input_%d' % (i - 1)
                    feature[imName] = tf.FixedLenFeature([], tf.string)
                parsed_example = tf.parse_single_example(example_proto, feature)
                patch = {}
                patch['gt'] = tf.image.decode_png(parsed_example['gt'], channels=3) / self.maxIntensity
                inputs = []
                for i in range(5):
                    imName = 'input_%d' % i
                    inputs.append(tf.image.decode_png(parsed_example[imName], channels=3) / self.maxIntensity)
                patch['input'] = tf.concat(inputs, -1)
                return patch

            datasetT = datasetRaw.map(parse_function, num_parallel_calls=6) \
                .repeat() \
                .shuffle(buffer_size=3200) \
                .prefetch(buffer_size=640) \
                .batch(batchSize) \
                .make_one_shot_iterator() \
                .get_next()

            return datasetT['input'], datasetT['gt']
