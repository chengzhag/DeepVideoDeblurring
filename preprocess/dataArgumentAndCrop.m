
%% Clean
clear
close all

%% Parameters
global batchSize cropWidth widthNeighbor
global frameExt nArguments argumentZoom
alignments = {'_nowarp'};%{'_nowarp','_OF','_homography'};
inputDir = '/home/zhangcheng/projects/DeepVideoDeblurring/data';
inputFolderPrefix = 'training_real_all_nostab';
gtDir = '/home/zhangcheng/projects/DeepVideoDeblurring/dataset/quantitative_datasets';

saveDir = '../data';
saveFolderPrefixTrain = 'training_augumented_all_nostab';
saveFolderPrefixValid = 'validating_augumented_all_nostab';

frameExt = '.jpg';
cropWidth = 128;
nCrop = 10;
nArguments = 2*4*4;
widthNeighbor = 5;
batchSize = 64;
currSpeed = 0;
argumentZoom = [1/4, 1/3, 1/2, 1];

%% Scan videos
inputFolders = {};
for alignment = alignments
    inputFolders{end+1} = [inputFolderPrefix,alignment{1}];
end
inputFolders = fullfile(inputDir,inputFolders);
videoNames = {};
for inputFolder = inputFolders
    videoNames{end+1} = dir(inputFolder{1});
    maskFolders = [videoNames{end}.isdir];
    videoNames{end} = videoNames{end}(maskFolders);
    videoNames{end} = videoNames{end}(3:end);
    videoNames{end} = {videoNames{end}.name};
end

validset = {'IMG_0030' 'IMG_0049' 'IMG_0021' '720p_240fps_2' 'IMG_0032' ...
'IMG_0033' 'IMG_0031' 'IMG_0003' 'IMG_0039' 'IMG_0037'};% from git rep DeepVideoDeburring

%% Generate batches
for iAlignment = 1:length(alignments)
    alignment = alignments{iAlignment};
    
    %% scan frames
    trainset = setdiff(videoNames{iAlignment},validset);
    frameDirs = scanFrames(inputFolders{iAlignment},gtDir,trainset);
    
    %% generate batches
    saveAlignFolder = fullfile(saveDir,[saveFolderPrefixTrain,alignment]);
    checkDir(saveAlignFolder);
    generateBatches(frameDirs,saveAlignFolder,nCrop*length(frameDirs)/batchSize*nArguments)
    
    %% scan validset frames
    frameDirs = scanFrames(inputFolders{iAlignment},gtDir,validset);
    
    %% generate validset batches
    saveAlignFolder = fullfile(saveDir,[saveFolderPrefixValid,alignment]);
    checkDir(saveAlignFolder);
    generateBatches(frameDirs,saveAlignFolder,nCrop*length(frameDirs)/batchSize*nArguments)
end


%% functions 

function frameDirs = scanFrames(inputFolder,gtDir,videoNames)
global frameExt widthNeighbor
% for all videos
disp('Scanning frames...');
frameDirs = {};
iFrameAll = 1;
for iVideo = 1:length(videoNames)
    videoName = videoNames{iVideo};
    inputFrameFolder = fullfile(inputFolder,videoName);
    inputFrameNames = dir(fullfile(inputFrameFolder,'image_0',['*',frameExt]));
    inputFrameNames = {inputFrameNames.name};
    GTFrameFolder = fullfile(gtDir,videoName,'GT');
    GTFrameNames = dir(fullfile(GTFrameFolder,['*',frameExt]));
    GTFrameNames = {GTFrameNames.name};
    for iFrame = 1:length(inputFrameNames)
        neighborDirs = cell(widthNeighbor,1);
        for iNeighbor = 1:widthNeighbor
            iFrameNeighbor = iNeighbor-ceil(widthNeighbor/2);
            neighborDirs{iNeighbor} = [inputFrameFolder,'/',sprintf('image_%d',iFrameNeighbor),'/',inputFrameNames{iFrame}];
        end
        frameDirs{iFrameAll,1} = neighborDirs;
        frameDirs{iFrameAll,2} = [GTFrameFolder,'/',GTFrameNames{iFrame}];
        iFrameAll = iFrameAll+1;
    end
    fprintf('Found %d frames, video %d/%d\n',iFrameAll-1,iVideo,length(videoNames))
end
fprintf('Found %d frames\n',iFrameAll-1)
end

function generateBatches(frameDirs,saveFolder,nBatches)
global batchSize cropWidth widthNeighbor argumentZoom
% scan for breakpoint
batchesDone = dir(fullfile(saveFolder,'*.mat'));
iBatchStart = 1;
if length(batchesDone) > 0
    iBatchStart = sscanf(batchesDone(end).name,'batch_width%d_size%d_%05d.mat');
    iBatchStart = iBatchStart(end)+2;
end
% start generating
tic;
for iBatch = iBatchStart:nBatches 
    isFrames = randi(length(frameDirs),batchSize,1);
    batchInput = zeros(batchSize,cropWidth,cropWidth,widthNeighbor*3,'uint8');
    batchGT = zeros(batchSize,cropWidth,cropWidth,3,'uint8');
    parfor iPatch = 1:batchSize
        zoom = argumentZoom(unidrnd(4));
        flip = rand>0.5;
        rotate = unidrnd(4);
        
        inputDirs = frameDirs{isFrames(iPatch),1};
        GTDir = frameDirs{isFrames(iPatch),2};
        gtIm = imread(GTDir);
        gtIm = argument(gtIm,zoom,flip,rotate);
        [h,w,c] = size(gtIm);
        patchUncroped = zeros(h,w,c,widthNeighbor);
        patchUncroped(:,:,:,1) = gtIm;
        for iNeighbor = 1:widthNeighbor
            inputIm = imread(inputDirs{iNeighbor});
            patchUncroped(:,:,:,iNeighbor+1) = argument(inputIm,zoom,flip,rotate);
        end
        patchCroped = randomCrop(patchUncroped,cropWidth);
%         montage(patchCroped/255);
%         pause(1)
        batchInput(iPatch,:,:,:) = reshape(patchCroped(:,:,:,2:end),[cropWidth,cropWidth,3*widthNeighbor]);
        batchGT(iPatch,:,:,:) = squeeze(patchCroped(:,:,:,1));
%         imshow(squeeze(batchInput(iPatch,:,:,1:3)));
%         pause(1)
    end
    batchDir = fullfile(saveFolder,sprintf( ...
        'batch_width%d_size%d_%05d.mat',cropWidth,batchSize,iBatch-1));
    disp(['saving to ' batchDir]);
    batchInputTorch = permute(batchInput,[1,4,2,3]);
    batchGTTorch = permute(batchGT,[1,4,2,3]);
    save(batchDir,'batchInputTorch','batchGTTorch','-v6');
    ms = toc*(nBatches-iBatch)/60;
    tic;
    hours = floor(ms/60);
    mins = mod(ms,60);
    fprintf('Generating... %.2f%%, %d hours %.1f minutes left.\n', ...
    iBatch/nBatches*100,hours,mins);
end
end

function imsCrop = randomCrop(ims,cropWidth)
[h,w,~,~] = size(ims);
wRange = w-cropWidth+1;
hRange = h-cropWidth+1;
wStart = randi(wRange);
hStart = randi(hRange);
isHCrop = hStart:hStart+cropWidth-1;
isWCrop = wStart:wStart+cropWidth-1;
imsCrop = ims(isHCrop,isWCrop,:,:);
end

function imArg = argument(im,zoom,flip,rotate)
imArg = imresize(im, zoom);
if flip
    imArg = fliplr(imArg);
end
imArg = rot90(imArg,rotate);
end
