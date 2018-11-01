require 'xlua'

opt = lapp[[
    --seed                  (default 251)                rand seed

    --model                 (default 'model')            model name
    --model_param_load      (default '')                 weight file to load
    --bn_meanstd_load       (default '')                 batch normalization file to load
    --optimstate_load       (default '')                 state of optim.adam to load

    --num_frames            (default 5)                  number of frames in input stack
    --num_channels          (default 3)                  rgb input
    --max_intensity         (default 255)                maximum intensity of input

    --data_root             (default 'data/training')    folder for traning data
    --trainset_size         (default '61')               size of trainset (if is larger than dataset, use all dataset as trainset)
    --batch_size            (default '64')               size of batch sampled from batch files (if is larger than size from batch files, use entire batch every iteration)
    --it_max                (default 80000)              max number of iterations

    --model_param           (default 'param')            weight file
    --bn_meanstd            (default 'bn_meanvar')       batch normalization file
    --optimstate            (default 'optimstate')       state of optim.adam
    --save_every            (default '500')              auto save every save_every iterations
    --log                   (default '')                dir to save log
]]
    -- log_every             (default 10)                 log every log_every iterations

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


-- load model
require 'cunn'
require 'cutorch'
require 'nn'
require 'optim'

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
itMax = opt.it_max
lrMin = 1e-6

print(string.format( "Params setting: max_intensity = %f",max_intensity ))
print(string.format( "                    batchSize = %d",batchSize ))
print(string.format( "                    decayFrom = %d",decayFrom ))
print(string.format( "                   decayEvery = %d",decayEvery ))
print(string.format( "                    decayRate = %f",decayRate ))
print(string.format( "                        itMax = %d",itMax ))
print(string.format( "                        lrMin = %f",lrMin ))

criterion = nn.MSECriterion()
criterion = criterion:cuda()

params, gradParams = net:getParameters()

paramsSaveDir = opt.model_param
bn_meanvarSaveDir = opt.bn_meanstd
optimstateSaveDir = opt.optimstate
saveEvery = opt.save_every


-- load params
paramsLoadDir = opt.model_param_load
bn_meanvarLoadDir = opt.bn_meanstd_load
optimstateLoadDir = opt.optimstate_load

function loadParams()
    print(string.format(
    "Loading params from: \nparamsLoadDir: %s \nbn_meanvarLoadDir: %s \noptimstateLoadDir: %s ", 
    paramsLoadDir,bn_meanvarLoadDir,optimstateLoadDir))
    if not pcall(
            function()
                local paramsSave = torch.load(paramsLoadDir)
                local bn_meanvarSave = torch.load(bn_meanvarLoadDir)
                optimstateSave = torch.load(optimstateLoadDir)

                params:copy(paramsSave)
                local bn_mean, bn_std = table.unpack(bn_meanvarSave)
                for k,v in pairs(net:findModules('nn.SpatialBatchNormalization')) do
                    v.running_mean:copy(bn_mean[k])
                    v.running_var:copy(bn_std[k])
                end
            end) 
    then
        print('No params to load or wrong dir!\n')
    end
    
    optimConfig = optimstateSave or {}
    optimConfig.learningRate = optimConfig.learningRate or 0.005
    optimConfig.weightDecay = optimConfig.weightDecay or 0
    optimConfig.beta1 = optimConfig.beta1 or 0.9
    optimConfig.beta2 = optimConfig.beta2 or 0.999
    optimConfig.epsilon = optimConfig.epsilon or 1e-8
    print('optimConfig = ')
    print(optimConfig)
    optimConfig.t = optimConfig.t or 0
    itBegin = optimConfig.t+1
end

loadParams()


-- load random batch sample
matio = require 'matio'
local function loadRandomBatchFrom(videoNames,batchSize)
    local iVideo = math.random(#videoNames)
    local iBatch = math.random(#batchDirs[videoNames[iVideo] ])
    local sampleDir = batchDirs[videoNames[iVideo] ][iBatch]
    local batchSample = matio.load(sampleDir)

    local batchInputRaw,batchGTRaw = batchSample.batchInputTorch, batchSample.batchGTTorch

    batchSize = batchSize or batchInputRaw:size(1)
    if batchSize - batchInputRaw:size(1) ~= 0 then
        local shuffle = torch.randperm(batchInputRaw:size(1))
        batchInput = torch.zeros(batchSize,batchInputRaw:size(2),batchInputRaw:size(3),batchInputRaw:size(4))
        batchGT = torch.zeros(batchSize,batchGTRaw:size(2),batchGTRaw:size(3),batchGTRaw:size(4))
        for i=1,batchSize do
            batchInput[i] = batchInputRaw[shuffle[i] ]
            batchGT[i] = batchGTRaw[shuffle[i] ]
        end
    else
        batchInput = batchInputRaw
        batchGT = batchGTRaw
    end

    return batchInput, batchGT
end

-- iteration funtion
local function feval(params)
    gradParams:zero()

    local batchInput, batchGT = loadRandomBatchFrom(trainset,batchSize)

    batchInput = batchInput:float():div(max_intensity):cuda()
    batchGT = batchGT:float():div(max_intensity):cuda()
        
    local outputs = net:forward(batchInput)
    loss = criterion:forward(outputs, batchGT)
    local dloss_doutputs = criterion:backward(outputs,batchGT)
    net:backward(batchInput, dloss_doutputs)
        
    return loss, gradParams
end


-- save params
local function saveParams(it)
    local paramsSaveDir = paramsSaveDir..'_iteration_'..it..'.t7'
    local bn_meanvarSaveDir = bn_meanvarSaveDir..'_iteration_'..it..'.t7'
    local optimstateSaveDir = optimstateSaveDir..'_iteration_'..it..'.t7'
    print(string.format(
        "Saving params to: \nparamsSaveDir: %s \nbn_meanvarSaveDir: %s \noptimstateSaveDir: %s ", 
        paramsSaveDir,bn_meanvarSaveDir,optimstateSaveDir))
    torch.save(paramsSaveDir,params)
    local bn_mean = {}
    local bn_std = {}
    for k,v in pairs(net:findModules('nn.SpatialBatchNormalization')) do
        table.insert(bn_mean,v.running_mean)
        table.insert(bn_std,v.running_var)
    end
    local bn_meanvar = {bn_mean,bn_std}
    torch.save(bn_meanvarSaveDir,bn_meanvar)
    torch.save(optimstateSaveDir,optimConfig)
    print(optimConfig)
end


-- Log results to files
if opt.log ~= '' then
    print('logging to: '..opt.log)
    trainLogger = optim.Logger(opt.log)
    trainLogger:setNames{'loss'}
end

-- train
-- at least 12 GB video memory required for batch size 64
math.randomseed(sys.clock()) 
local tic = sys.clock()
net:training() -- set train = true
for it = itBegin,itMax do
    if it>=decayFrom and (it-decayFrom)%decayEvery==0 then
        optimConfig.learningRate = optimConfig.learningRate*decayRate
    end
    if optimConfig.learningRate < lrMin then
        optimConfig.learningRate = lrMin
    end

    optim.adam(feval,params,optimConfig)
    
    local toc = sys.clock()
    speed = 1/(toc-tic)
    avgSpeed = avgSpeed or speed
    avgSpeed = 0.1*speed+0.9*avgSpeed
    timeLeft = (itMax-it)/avgSpeed
    toc = tic
    tic = sys.clock()
    timeLeftMin = timeLeft/60
    print(string.format(
            'it: %d, loss: %f, lr: %f, spd: %.2f it/s, avgSpd: %.2f it/s, left: %d h %.1f min',
            it,loss,optimConfig.learningRate,speed,avgSpeed,timeLeftMin/60,timeLeftMin%60))
    
    -- update logger/plot
    if trainLogger then
        trainLogger:add{loss}
        trainLogger:style{'-'}
        trainLogger:plot()
    end

    if it%opt.save_every == 0 or it == itMax then 
        saveParams(it)
    end
end





