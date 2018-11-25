
%% Clean
clear
close all

%% Parameters
global batchSize cropWidth widthNeighbor
global frameExt nArguments
alignments = {'_nowarp'};%,'_OF','_homography'};
inputDir = '../data';
inputFolderPrefix = 'training_augumented_uncroped_all_nostab';
saveDir = 'E:\projects\DeepVideoDeblurring\data';
saveFolderPrefixTrain = 'training_augumented_croped_all_nostab';
saveFolderPrefixValid = 'validating_augumented_croped_all_nostab';
frameExt = '.jpg';
cropWidth = 128;
nCrop = 10;
nArguments = 2*4*4;
widthNeighbor = 5;
batchSize = 64;
currSpeed = 0;

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
    frameDirs = scanFrames(inputFolders{iAlignment},trainset);
    
    %% generate batches
    saveAlignFolder = fullfile(saveDir,[saveFolderPrefixTrain,alignment]);
    checkDir(saveAlignFolder);
    generateBatches(frameDirs,saveAlignFolder,nCrop*length(frameDirs)/batchSize)
    
    %% scan validset frames
    frameDirs = scanFrames(inputFolders{iAlignment},validset);
    
    %% generate validset batches
    saveAlignFolder = fullfile(saveDir,[saveFolderPrefixValid,alignment]);
    checkDir(saveAlignFolder);
    generateBatches(frameDirs,saveAlignFolder,nCrop*length(frameDirs)/batchSize)
end


%% functions 
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

function frameDirs = scanFrames(inputFolders,videoNames)
global frameExt widthNeighbor nArguments
% for all videos
disp('Scanning frames...');
frameDirs = {};
iFrameAll = 1;
for iVideo = 1:length(videoNames)
    videoName = videoNames{iVideo};
    inputFrameFolder = fullfile(inputFolders,videoName,'input');
    inputFrameNames = dir(fullfile(inputFrameFolder,['*',frameExt]));
    inputFrameNames = {inputFrameNames.name};
    GTFrameFolder = fullfile(inputFolders,videoName,'GT');
    GTFrameNames = dir(fullfile(GTFrameFolder,['*',frameExt]));
    GTFrameNames = {GTFrameNames.name};
    for iFrame = 1:length(inputFrameNames)
        neighborDirs = cell(widthNeighbor,1);
        for iNeighbor = 1:widthNeighbor
            iFrameNeighbor = ceil(iFrame/nArguments)+iNeighbor-ceil(widthNeighbor/2);
            iFrameNeighbor = max(1,min(iFrameNeighbor,length(inputFrameNames)/nArguments));
            iFrameNeighbor = (iFrameNeighbor-1)*nArguments+mod(iFrame-1,nArguments)+1;
            neighborDirs{iNeighbor} = [inputFrameFolder,'/',inputFrameNames{iFrameNeighbor}];
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
global batchSize cropWidth widthNeighbor
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
    for iPatch = 1:batchSize
        inputDirs = frameDirs{isFrames(iPatch),1};
        GTDirs = frameDirs{isFrames(iPatch),2};
        gtIm = imread(GTDirs);
        [h,w,c] = size(gtIm);
        patchUncroped = zeros(h,w,c,widthNeighbor);
        patchUncroped(:,:,:,1) = gtIm;
        for iNeighbor = 1:widthNeighbor
            patchUncroped(:,:,:,iNeighbor+1)=imread(inputDirs{iNeighbor});
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
