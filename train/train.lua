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
    --batch_size            (default '64')               size of batch sampled from batch files (if is larger than size from batch files, use entire batch every iteration)

    --save_every            (default '500')              auto save every save_every iterations
    --load_save             (default 0)                  if load from auto saved files
    --log_save              (default 'log.log')          dir to save log
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

math.randomseed(opt.seed) 
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

require 'cunn'
require 'cutorch'
require 'nn'
require 'optim'

-- load model
modelDir = opt.model
print(string.format( "Loading model %s",modelDir ))
require(modelDir)
net = create_model()
net = net:cuda()
print("Model loaded")


-- params setting
max_intensity = opt.max_intensity
batchSize = opt.batch_size
decayFrom = 24000
decayEvery = 8000
decayRate = 0.5
itMax = 80000
lossThres = 10e-6

print(string.format( "Params setting: max_intensity = %f",max_intensity ))
print(string.format( "                    batchSize = %d",batchSize ))
print(string.format( "                    decayFrom = %d",decayFrom ))
print(string.format( "                   decayEvery = %d",decayEvery ))
print(string.format( "                    decayRate = %f",decayRate ))
print(string.format( "                        itMax = %d",itMax ))
print(string.format( "                    lossThres = %f",lossThres ))

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

paramsSaveDir = opt.model_param
bn_meanvarSaveDir = opt.bn_meanstd
saveEvery = opt.save_every


-- load params
-- TODO: parse auto saved file name to find last break point
-- if opt.load_save == 1 then
--     -- and number of iterations done
--     print(string.format("Loading breakpoint from %s and %s ... ",paramsSaveDir,bn_meanvarSaveDir))
--     local paramsSave = torch.load(paramsSaveDir)
--     local bn_meanvarSave = torch.load(bn_meanvarSaveDir)

--     params:copy(paramsSave)
--     local bn_mean, bn_std = table.unpack(bn_meanvarSave)
--     for k,v in pairs(net:findModules('nn.SpatialBatchNormalization')) do
--         v.running_mean:copy(bn_mean[k])
--         v.running_var:copy(bn_std[k])
--     end
-- end


-- train net

-- load random batch sample
matio = require 'matio'
function loadRandomBatchFrom(videoNames)
    local iVideo = math.random(#videoNames)
    local iBatch = math.random(#batchDirs[videoNames[iVideo] ])
    local sampleDir = batchDirs[videoNames[iVideo] ][iBatch]
    local batchSample = matio.load(sampleDir)
    return batchSample.batchInputTorch, batchSample.batchGTTorch
end

-- iteration funtion
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
        
    return loss, gradParams
end

-- save params
-- TODO: save optim state
local function saveParams(paramsSave,bn_meanvarSave,it)
    local paramsSaveDir = paramsSave..'_iteration_'..it..'.t7'
    local bn_meanvarSaveDir = bn_meanvarSave..'iteration_'..it..'.t7'
    print(string.format( "Saving params to %s and %s",paramsSaveDir,bn_meanvarSaveDir ))
    torch.save(paramsSaveDir,params)
    local bn_mean = {}
    local bn_std = {}
    for k,v in pairs(net:findModules('nn.SpatialBatchNormalization')) do
        table.insert(bn_mean,v.running_mean)
        table.insert(bn_std,v.running_var)
    end
    local bn_meanvar = {bn_mean,bn_std}
    torch.save(bn_meanvarSaveDir,bn_meanvar)
end

-- Log results to files
-- TODO: append if train from a breakpoint
trainLogger = optim.Logger(opt.log_save)

-- at least 12 GB video memory required for batch size 64
math.randomseed(opt.seed) 
local tic = sys.clock()
local time = tic;
net:training() -- set train = true
for it = 1,itMax do
    if it>=decayFrom and (it-decayFrom)%decayEvery==0 then
        optimConfig.learningRate = optimConfig.learningRate*decayRate;
    end

    optim.sgd(feval,params,optimState)
    
    local toc = sys.clock()
    timeLeft = (toc-tic)/it*(itMax-it)
    
    print(string.format(
            'it: %d, loss: %f, lr: %f, left: %.2fmin',
            it,loss,optimConfig.learningRate,timeLeft/60))
    
    -- TODO: learn more about optim.Logger
    -- update logger/plot
    trainLogger:add{['loss'] = loss}
--     trainLogger:style{['loss'] = '-'}
--     trainLogger:plot()
        
    if loss<lossThres then
        saveParams(paramsSaveDir,bn_meanvarSaveDir,it)
        break
    end

    if it%opt.save_every == 0 or it == itMax then 
        saveParams(paramsSaveDir,bn_meanvarSaveDir,it)
    end
end





