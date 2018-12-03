require 'xlua'

opt = lapp[[
    --seed                  (default 251)                rand seed

    --model                 (default '')                 model name
    --model_param_load      (default '')                 weight file to load
    --bn_meanstd_load       (default '')                 batch normalization file to load
    --optimstate_load       (default '')                 state of optim.adam to load

    --num_frames            (default 5)                  number of frames in input stack
    --num_channels          (default 3)                  rgb input
    --max_intensity         (default 255)                maximum intensity of input

    --data_root             (default '')                 folder for datasets
    --data_trainset         (default '')                 folder for transet data
    --data_validset         (default '')                 folder for validset data
    --trainset_size         (default 61)                 size of trainset (if is larger than dataset, use all dataset as trainset)
    --batch_size            (default 64)                 size of batch sampled from batch files (if is larger than size from batch files, use entire batch every iteration)
    --it_max                (default 80000)              max number of iterations

    --model_param           (default '')                 weight file
    --bn_meanstd            (default '')                 batch normalization file
    --optimstate            (default '')                 state of optim.adam
    --save_every            (default 500)                auto save every save_every iterations
    --log                   (default '')                 dir to save log
    --log_every             (default 1)                  log every log_every iterations

    --reset_lr              (default 0)                  reset lr to reset_lr instead of default or saved lr
    --reset_state           (default 0)                  reset optim state
    --decay_from            (default 24000)              decay learning rate from decay_from iterations
    --decay_every           (default 8000)               decay learning rate every decay_every iteration
    --decay_rate            (default 0.5)                decay learning rate

    --overfit_batches       (default 0)                  overfit batch num for debuging
    --overfit_patches       (default 0)                  overfit patch num, valid if overfit_batches = 1
    --overfit_out           (default '')                 overfit output on trainset
]]
    

-- scan batches
require("lfs")
trainsetDirs = {}
validsetDirs = {}
if #opt.data_root > 0 then

    alignFolder = opt.data_root
    print(string.format('Scanning videos from %s',alignFolder))
    videoNames = {}
    videoName2batchDirs = {}
    local iVideos = 0;
    local nBatches = 0;
    for videoName in lfs.dir(alignFolder) do
        if videoName ~= "." and videoName ~= ".." then
    --         print(videoName)
            iVideos = iVideos+1
            videoNames[iVideos] = videoName
            videoName2batchDirs[videoName] = {};
            local videoFolder = paths.concat(alignFolder,videoName);
            for batchName in lfs.dir(videoFolder) do
                if batchName ~= "." and batchName ~= ".." then
                    table.insert(videoName2batchDirs[videoName],
                        paths.concat(videoFolder,batchName))
    --                 print(batchName)
                end
            end
            table.sort(videoName2batchDirs[videoName])
            nBatches = nBatches + #videoName2batchDirs[videoName]
        end
    end
    table.sort(videoNames)
    print(string.format('Found %d videos and %d batches',#videoNames,nBatches))

    -- split datasets (videoNames) into trainset and validset
    local function getDirsFromNames(names)
        local dirs = {}
        for iName,name in ipairs(names) do
            local dirsName = videoName2batchDirs[name]
            for iDir,dir in ipairs(dirsName) do
                table.insert(dirs,dir)
            end
        end
        return dirs
    end

    math.randomseed(opt.seed) 
    if opt.overfit_batches == 0 then
        local trainsetSize = math.min(opt.trainset_size,#videoNames)

        local name2weight = {}
        for iVideo,videoName in ipairs(videoNames) do
            name2weight[videoName] = math.random()
        end
        table.sort(videoNames,function(a,b) return name2weight[a]<name2weight[b] end)
        trainsetNames={}
        validsetNames={}
        for iVideo,videoName in ipairs(videoNames) do
            if(iVideo<=trainsetSize) then
                table.insert(trainsetNames,videoName)
            else
                table.insert(validsetNames,videoName)
            end
        end
        print(string.format('%d for trainset and %d for validset',#trainsetNames,#validsetNames))

        trainsetDirs = getDirsFromNames(trainsetNames)
        validsetDirs = getDirsFromNames(validsetNames)
        table.sort(trainsetDirs)
        table.sort(validsetDirs)
        print(string.format('Split batches into %d for trainset and %d for validset',#trainsetDirs,#validsetDirs))
    else
        print(string.format( "Overfitting %d batches",opt.overfit_batches ))
        local datasetDirs = getDirsFromNames(videoNames)
        for iBatch = 1,opt.overfit_batches do
            table.insert(trainsetDirs,datasetDirs[iBatch])
        end
    end

elseif #opt.data_trainset > 0 and #opt.data_validset > 0 then

    trainsetFolder = opt.data_trainset
    print(string.format('Scanning trainset batches from %s',trainsetFolder))
    for batchDir in lfs.dir(trainsetFolder) do
        if batchDir ~= "." and batchDir ~= ".." then
            table.insert(trainsetDirs,paths.concat(trainsetFolder,batchDir));
        end
    end
    table.sort(trainsetDirs)
    print(string.format('Found %d trainset batches',#trainsetDirs))

    validsetFolder = opt.data_validset
    print(string.format('Scanning validset batches from %s',validsetFolder))
    
    for batchDir in lfs.dir(validsetFolder) do
        if batchDir ~= "." and batchDir ~= ".." then
            table.insert(validsetDirs,paths.concat(validsetFolder,batchDir));
        end
    end
    table.sort(validsetDirs)
    print(string.format('Found %d validset batches',#validsetDirs))

    if opt.overfit_batches > 0 then
        print(string.format( "Overfitting %d batches",opt.overfit_batches ))
        trainsetDirsTemp = {}
        for iBatch = 1,opt.overfit_batches do
            table.insert(trainsetDirsTemp,trainsetDirs[iBatch])
        end
        trainsetDirs = trainsetDirsTemp
    end
end

local eampleBatchNum = math.min(10,#trainsetDirs)
print(string.format( "First %d batches example of %d trainset Batch:",eampleBatchNum,#trainsetDirs ))
for iBatch = 1,eampleBatchNum do
    print('\t'..trainsetDirs[iBatch])
end
local eampleBatchNum = math.min(10,#validsetDirs)
print(string.format( "First %d batches example of %d validset Batch:",eampleBatchNum,#validsetDirs ))
for iBatch = 1,eampleBatchNum do
    print('\t'..validsetDirs[iBatch])
end


-- preload trainset if it's small
trainsetBatches = {}
if #trainsetDirs <= 100 then
    matio = require 'matio'
    print('Preloading '..#trainsetDirs..' trainset into memory')
    for iBatch,trainsetDir in ipairs(trainsetDirs) do
        table.insert(trainsetBatches, matio.load(trainsetDir))
        print(iBatch..'/'..#trainsetDirs)
    end
end


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
decayFrom = opt.decay_from
decayEvery = opt.decay_every
decayRate = opt.decay_rate
lrMin = 1e-6
itMax = opt.it_max

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
                if opt.reset_state==0 then
                    optimstateSave = torch.load(optimstateLoadDir)
                end
                assert(params:nElement() == paramsSave:nElement(), string.format('%s: %d vs %d', 'loading parameters: dimension mismatch.', params:nElement(), paramsSave:nElement()))
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
    optimConfig.learningRate = optimConfig.learningRate or 0.005 --default lr=0.005
    if opt.reset_lr > 0 then
        optimConfig.learningRate = opt.reset_lr
    end
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
function copyBatch(batch)
    local batchCopy = {}
    batchCopy.batchInputTorch = batch.batchInputTorch:clone()
    batchCopy.batchGTTorch = batch.batchGTTorch:clone()
    return batchCopy
end
matio = require 'matio'
function loadRandomBatchFrom(batchDirs,batchSize)
    local iBatch = math.random(#batchDirs)
    local batchSample = nil
    if type(batchDirs[1]) == 'string' then
        local sampleDir = batchDirs[iBatch]
        batchSample = matio.load(sampleDir)
    else
        batchSample = copyBatch(batchDirs[iBatch])
    end

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


-- save params
local function saveParams(it)
    if paramsSaveDir ~= '' and bn_meanvarSaveDir ~= '' and optimstateSaveDir ~= '' then
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
    else
        print('No dir to save!\n')
    end
end


-- preload batch if overfit_batches = 1
if opt.overfit_batches == 1 then
    batchInput, batchGT = loadRandomBatchFrom(trainsetDirs,batchSize)
    if opt.overfit_batches == 1 and opt.overfit_patches > 0 then
        batchInput = batchInput[{{1,opt.overfit_patches},{},{},{}}]
        batchGT = batchGT[{{1,opt.overfit_patches},{},{},{}}]
    end
    batchInput = batchInput:double():div(max_intensity):cuda()
    batchGT = batchGT:double():div(max_intensity):cuda()
end


-- iteration funtion
local function feval(params)
    gradParams:zero()
    if #trainsetBatches > 0 then
        batchInput, batchGT = loadRandomBatchFrom(trainsetBatches,batchSize)
    else
        batchInput, batchGT = loadRandomBatchFrom(trainsetDirs,batchSize)
    end
    batchInput = batchInput:double():div(max_intensity):cuda()
    batchGT = batchGT:double():div(max_intensity):cuda()

    local batchOutput = net:forward(batchInput)
    loss = criterion:forward(batchOutput, batchGT)
    local dloss_dbatchOutput = criterion:backward(batchOutput,batchGT)
    net:backward(batchInput, dloss_dbatchOutput)
        
    return loss, gradParams
end


-- train
-- Log results to files
if opt.log ~= '' then
    paths.mkdir(string.match(opt.log, "(.+)/[^/]*%.%w+$"))
    print('logging to: '..opt.log)
    trainLogger = optim.Logger(opt.log)
    trainLogger:setNames{'loss'}
end
-- at least 12 GB video memory required for batch size 64
math.randomseed(sys.clock()) 
net:training() -- set train = true
local tic = sys.clock()
lossAvg = 0;
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
    lossAvg = lossAvg + loss
    -- update logger/plot
    if trainLogger and it%opt.log_every == 0 then
        lossAvg = lossAvg/opt.log_every
        trainLogger:add{lossAvg}
        trainLogger:style{'-'}
        trainLogger:plot()
        lossAvg = 0
    end

    if it%opt.save_every == 0 or it == itMax then 
        saveParams(it)
    end
end


-- test on trainset for overfitting
require 'image'
if opt.overfit_batches > 0 then
    print('Testing on trainset for overfitting')
    local outImgSaveDir =  opt.overfit_out
    local outputSaveDir = paths.concat(outImgSaveDir,'output')
    local gtSaveDir = paths.concat(outImgSaveDir,'GT')
    local inputSaveDir = paths.concat(outImgSaveDir,'input')
    paths.mkdir(outputSaveDir)
    paths.mkdir(gtSaveDir)
    paths.mkdir(inputSaveDir)

    print(string.format("Saving test result to: %s",outImgSaveDir))
    net:evaluate() -- get different loss?
    local lossTest = 0
    local maxLossTest = 0
    local tic = sys.clock()
    for iBatch,trainsetDir in ipairs(trainsetDirs) do
        local batchSample = nil
        if #trainsetBatches > 0 then
            batchSample = copyBatch(trainsetBatches[iBatch])
        else
            batchSample = matio.load(trainsetDir)
        end
        local batchInput,batchGT = batchSample.batchInputTorch, batchSample.batchGTTorch
        if opt.overfit_batches == 1 and opt.overfit_patches > 0 then
            batchInput = batchInput[{{1,opt.overfit_patches},{},{},{}}]
            batchGT = batchGT[{{1,opt.overfit_patches},{},{},{}}]
        end
        batchInput = batchInput:double():div(max_intensity):cuda()
        batchGT = batchGT:double():div(max_intensity):cuda()

        local batchOutput = net:forward(batchInput)
        local loss = criterion:forward(batchOutput, batchGT)
        maxLossTest = math.max(maxLossTest,loss)
        lossTest = lossTest+loss

        batchOutput:mul(max_intensity)
        batchGT:mul(max_intensity)
        batchInput = batchInput:contiguous():view(batchInput:size(1),5,3,128,128)
        batchInput:mul(max_intensity)
        for iPatch = 1,batchOutput:size(1) do
            imName = string.match(trainsetDir,".+/([^/]*)%.%w+$")
            image.save(paths.concat(outputSaveDir,string.format('%s_%02d.jpg',imName,iPatch-1)), batchOutput[iPatch]:byte())
            image.save(paths.concat(gtSaveDir,string.format('%s_%02d.jpg',imName,iPatch-1)), batchGT[iPatch]:byte())
            image.save(paths.concat(inputSaveDir,string.format('%s_%02d.jpg',imName,iPatch-1)), batchInput[iPatch][3]:byte())
        end

        local toc = sys.clock()
        speed = 1/(toc-tic)
        timeLeft = (#trainsetDirs-iBatch)/speed
        toc = tic
        tic = sys.clock()
        timeLeftMin = timeLeft/60
        print(string.format(
                'ba: %d/%d, loss: %f, spd: %.2f ba/s, left: %d h %.1f min',
                iBatch,#trainsetDirs,loss,speed,timeLeftMin/60,timeLeftMin%60))

    end
    lossTest = lossTest/#trainsetDirs
    print(string.format('Final test loss: %f, max test loss: %f',lossTest,maxLossTest))
end




