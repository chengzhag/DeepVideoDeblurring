
%% Clean
clear
close all

%% Parameters
global nAlignments nVideos processBar batchMB currSpeed;
global batchSize cropWidth widthNeighbor;
alignments = {'_nowarp'};%,'_OF','_homography'};
nAlignments = length(alignments);
inputDir = '../data';
gtDir = '../dataset/quantitative_datasets';
inputFolderPrefix = 'training_real_all_nostab';
breakpointFile = 'breakpoint_dataArgumentAndCrop.txt';
saveDir = '../data';
saveFolderPrefix = 'training_augumented_all_nostab';
frameExt = '.jpg';
argumentZoom = {1/4, 1/3, 1/2, 1};
cropWidth = 128;
nCrops = 10;
widthNeighbor = 5;
batchSize = 64;
batchMB = batchSize*cropWidth*cropWidth*3*(widthNeighbor+1)/1e6;
currSpeed = 0;
global iBatch iPatch;
iBatch = 0;
iPatch = 1;

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

%% Prepare
nVideos = length(videoNames{1});
% batch make
qPushPatch = parallel.pool.DataQueue;
lPushPatch = qPushPatch.afterEach(@(params) pushPatch(params));

%% Argument and crop
% for all alignments
tic;
for iAlignment = 1:nAlignments
    alignment = alignments{iAlignment};
    nVideos = length(videoNames{iAlignment});
    saveAlignFolder = fullfile(saveDir,[saveFolderPrefix,alignment]);
    checkDir(saveAlignFolder);
    % for all videos
    for iVideo = 1:nVideos
        videoName = videoNames{iAlignment}{iVideo};
        inputFrameNames = dir( ...
            fullfile(inputFolders{iAlignment},videoName,'image_0',['*',frameExt]));
        inputFrameNames = {inputFrameNames.name};
        GTFrameNames = dir(fullfile(gtDir,videoName,'GT',['*',frameExt]));
        GTFrameNames = {GTFrameNames.name};
        nFrames = length(inputFrameNames);
        
        % check dir
        saveVideoDir = fullfile(saveAlignFolder,videoName);
        disp(saveVideoDir);
        checkDir(saveVideoDir);
        
        % for all frames
        for iFrame = 1:nFrames
            frameName = inputFrameNames{iFrame};
            
            % argument
            frameDir = fullfile(gtDir,videoName,'GT',GTFrameNames{iFrame});
            disp(frameDir);
            gt = imread(frameDir);
            gtsArg = argument(gt,argumentZoom);
            inputsArg = cell(1,widthNeighbor);
            for iNeighbor1i = 1:widthNeighbor
                iNeighbor = iNeighbor1i-ceil(widthNeighbor/2);
                input = imread(fullfile( ...
                    inputFolders{iAlignment}, ...
                    videoName, ...
                    ['image_',num2str(iNeighbor)], ...
                    frameName));
                inputsArg{iNeighbor1i} = argument(input,argumentZoom);
            end
            % argument - end
            
            % crop
            nArguments = length(gtsArg);
            for iArgument=1:nArguments
                frames = inputsArg{1}{iArgument};
                [h,w,~] = size(frames);
                wRange = w-cropWidth+1;
                hRange = h-cropWidth+1;
                wStarts = randi(wRange,1,nCrops);
                hStarts = randi(hRange,1,nCrops);
                gt = gtsArg{iArgument};
                for iCrop = 1:nCrops
                    wStart = wStarts(iCrop);
                    hStart = hStarts(iCrop);
                    isHCrop = hStart:hStart+cropWidth-1;
                    isWCrop = wStart:wStart+cropWidth-1;

                    gtCrop = gt(isHCrop,isWCrop,:);

                    % crop patchInput
                    patchInput = zeros(cropWidth,cropWidth,3*widthNeighbor,'uint8');
                    for iNeighbor1i = 1:widthNeighbor
                        input = inputsArg{iNeighbor1i}{iArgument};
                        inputCrop = input(isHCrop,isWCrop,:);
                        patchInput(:,:,(iNeighbor1i-1)*3+1:(iNeighbor1i-1)*3+3) = inputCrop;
                    end
                    pushPatch(patchInput,gtCrop,saveVideoDir);
                end
            end
            % crop - end
            showProgress(1/nFrames);
        end
        % clear patch index -- start a new batch
        iPatch = 1;
    end
end
%% Clean

function imsArg = argument(im,argumentZoom)
imsArg = cell(2*4*length(argumentZoom));
iArg=1;
% argument - zoom
for zoom = argumentZoom
    imArg = imresize(im, zoom{1});
    % argument - flip
    for iFlip = 1:2
        % argument - rotate
        for iRotate = 1:4
            imsArg{iArg} = imArg;
%             imshow(imArg);
%             pause(0.1);
            imArg = rot90(imArg);
            iArg = iArg+1;
        end
        if iFlip == 2
            break;
        end
        imArg = fliplr(imArg);
    end
end
end

function showProgress(dProgress)
global nAlignments nVideos currSpeed;
persistent nVideoDone;
if isempty(nVideoDone)
    nVideoDone = 0;
end

nVideoDone = nVideoDone+dProgress;
x = nVideoDone/nAlignments/nVideos;
ms = toc/x*(1-x)/60;
hours = floor(ms/60);
mins = mod(ms,60);
fprintf('Generating... %.2f%%, %.2f MB/s, %d hours %.1f minutes left.', ...
    x*100,currSpeed,hours,mins);
end

function pushPatch(patchInput,patchGT,saveFolder)
global batchSize cropWidth widthNeighbor iBatch batchMB currSpeed iPatch;
persistent batchInput batchGT;
if isempty(batchInput)
    batchInput = zeros(batchSize,cropWidth,cropWidth,widthNeighbor*3,'uint8');
    batchGT = zeros(batchSize,cropWidth,cropWidth,3,'uint8');
end

batchInput(iPatch,:,:,:) = patchInput;
batchGT(iPatch,:,:,:) = patchGT;
iPatch = iPatch+1;
if iPatch > batchSize
    saveDir = fullfile(saveFolder,sprintf( ...
        'batch_width%d_size%d_%05d.mat',cropWidth,batchSize,iBatch));
    disp(['saving' saveDir]);
    save(saveDir,'batchInput','batchGT');
    currSpeed = batchMB*iBatch/toc;
    iPatch = 1;
    iBatch = iBatch+1;
end
end
