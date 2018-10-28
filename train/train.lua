require 'xlua'

opt = lapp[[
    --seed                  (default 251)                rand seed

    --model                 (default 'model')            model name
    --model_param           (default 'param')            weight file
    --bn_meanstd            (default 'bn_meanvar')       batch normalization file

    --num_frames            (default 5)                  number of frames in input stack
    --num_channels          (default 3)                  rgb input
    --max_intensity         (default 255)                maximum intensity of input

    --data_root             (default 'data/training')    folder for traning data
    --saveDir               (default 'logs')             folder for params and logs

    --trainset_size         (default '61')               size of trainset (if is larger than dataset, use all dataset as trainset)
]]

-- scan batches
require("lfs")

alignFolder = opt.data_root
print(alignFolder)
print(string.format('scanning videos from %s',alignFolder))
videoNames = {}
batchDirs = {}
local iVideos = 0;
local nBatches = 0;
for videoName in lfs.dir(alignFolder) do
    if videoName ~= "." and videoName ~= ".." then
--         print(videoName)
        iVideos = iVideos+1
        videoNames[iVideos] = videoName
        batchDirs[videoName] = {};
        local videoFolder = paths.concat(alignFolder,videoName);
        for batchName in lfs.dir(videoFolder) do
            if batchName ~= "." and batchName ~= ".." then
                table.insert(batchDirs[videoName],
                    paths.concat(videoFolder,batchName))
--                 print(batchName)
            end
        end
        table.sort(batchDirs[videoName])
        nBatches = nBatches + #batchDirs[videoName]
    end
end
table.sort(videoNames)
print(string.format('Found %d videos and %d batches',#videoNames,nBatches))


-- split datasets (videoNames) into trainset and validset
local trainsetSize = math.min(opt.trainset_size,#videoNames)

math.randomseed(251) 
local name2weight = {}
for iVideo,videoName in ipairs(videoNames) do
    name2weight[videoName] = math.random()
end
table.sort(videoNames,function(a,b) return name2weight[a]<name2weight[b] end)
trainset={}
validset={}
for iVideo,videoName in ipairs(videoNames) do
    if(iVideo<=trainsetSize) then
        table.insert(trainset,videoName)
    else
        table.insert(validset,videoName)
    end
end
print(string.format('Found %d trainsets and %d validsets',#trainset,#validset))

matio = require 'matio'
-- load random batch sample
function loadRandomBatchFrom(videoNames)
    local iVideo = math.random(#videoNames)
    local iBatch = math.random(#batchDirs[videoNames[iVideo]])
    local sampleDir = batchDirs[videoNames[iVideo]][iBatch]
    local batchSample = matio.load(sampleDir)
    return batchSample.batchInputTorch, batchSample.batchGTTorch
end

require 'cunn'
require 'cutorch'
require 'nn'
require 'optim'

-- load model
modelDir = opt.model
require(modelDir)
local batchInputSample,batchGTSample = loadRandomBatchFrom(trainset)
net = create_model(batchInputSample:size(2))
net = net:cuda()

--[[
-- params setting
max_intensity = 255
batchSize = 48
decayFrom = 24000
decayEvery = 8000
decayRate = 0.5
itMax = 20000
lossThres = 10e-6

criterion = nn.MSECriterion()
criterion = criterion:cuda()

params, gradParams = net:getParameters()
optimConfig = {
    learningRate = 0.005,
    weightDecay = 0,
    beta1 = 0.9,
    beta2 = 0.999,
    epsilon = 10e-8
}

paramsSaveDir = 'param.t7'
bn_meanvarSaveDir = 'bn_meanvar.t7'


-- load params
local paramsSave = torch.load(paramsSaveDir)
local bn_meanvarSave = torch.load(bn_meanvarSaveDir)

params:copy(paramsSave)
local bn_mean, bn_std = table.unpack(bn_meanvarSave)
for k,v in pairs(net:findModules('nn.SpatialBatchNormalization')) do
    v.running_mean:copy(bn_mean[k])
    v.running_var:copy(bn_std[k])
end

-- train net
-- Log results to files
trainLogger = optim.Logger('train.log')

-- at least 12 GB video memory required for batch size 64
math.randomseed(251) 
local tic = sys.clock()
local time = tic;
net:training() -- set train = true
for it = 1,itMax do
    if it>=decayFrom and (it-decayFrom)%decayEvery==0 then
        optimConfig.learningRate = optimConfig.learningRate*decayRate;
    end

    local function  feval(params)
        gradParams:zero()
    
        local batchInputRaw,batchGTRaw = loadRandomBatchFrom(trainset)
    
        local shuffle = torch.randperm(batchInputRaw:size(1))
        local batchInput = torch.zeros(batchSize,batchInputRaw:size(2),batchInputRaw:size(3),batchInputRaw:size(4))
        local batchGT = torch.zeros(batchSize,batchGTRaw:size(2),batchGTRaw:size(3),batchGTRaw:size(4))
        for i=1,batchSize do
            batchInput[i] = batchInputRaw[shuffle[i] ]
            batchGT[i] = batchGTRaw[shuffle[i] ]
        end
        batchInput = batchInput:float():div(max_intensity):cuda()
        batchGT = batchGT:float():div(max_intensity):cuda()
            
        local predict = net:forward(batchInput)
        loss = criterion:forward(predict, batchGT)
        local dloss_dpredict = criterion:backward(predict,batchGT)
        local gradInput = net:backward(trainset, dloss_dpredict)
        
    --     print(string.format('loss %f',loss))
            
        return loss, gradParams
    end
    optim.sgd(feval,params,optimState)
    
    local toc = sys.clock()
    timeLeft = (toc-tic)/it*(itMax-it)
    
    print(string.format(
            'it: %d, loss: %f, lr: %f, left: %.2fmin\n',
            it,loss,optimConfig.learningRate,timeLeft/60))
    
    -- update logger/plot
    trainLogger:add{['loss'] = loss}
--     trainLogger:style{['loss'] = '-'}
--     trainLogger:plot()
        
    if loss<lossThres then
        break
    end
end


-- save params
torch.save(paramsSaveDir,params)
local bn_mean = {}
local bn_std = {}
for k,v in pairs(net:findModules('nn.SpatialBatchNormalization')) do
    table.insert(bn_mean,v.running_mean)
    table.insert(bn_std,v.running_var)
end
local bn_meanvar = {bn_mean,bn_std}
torch.save(bn_meanvarSaveDir,bn_meanvar)
]]

