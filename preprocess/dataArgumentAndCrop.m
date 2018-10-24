
%% Clean
clear
close all

%% Parameters
global nDone nAlignments nVideos processBar;
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
nDone = 0;
processBar = waitbar(0,'Generating... ');
qUpdateWaitbar = parallel.pool.DataQueue;
lUpdateWaitbar = qUpdateWaitbar.afterEach(@(progress) updateWaitbar(progress));

%% Argument and crop
% for all alignments
tic;
for iAlignment = 1:nAlignments
    alignment = alignments{iAlignment};
    nVideos = length(videoNames{iAlignment});
    % for all videos
    parfor iVideo = 1:nVideos
        videoName = videoNames{iAlignment}{iVideo};
        frameNames = dir( ...
            fullfile(inputFolders{iAlignment},videoName,'image_0',['*',frameExt]));
        frameNames = {frameNames.name};
        nFrames = length(frameNames);
        
        % check dir
        saveVideoDir = fullfile(saveDir,[saveFolderPrefix,alignment],videoName);
        disp(saveVideoDir);
        gtCropFolder = fullfile(saveVideoDir,'GT');
        checkDir(gtCropFolder)
        inputCropFolders = cell(1,widthNeighbor);
        for iNeighbor1i = 1:widthNeighbor
            iNeighbor = iNeighbor1i-ceil(widthNeighbor/2);
            inputCropFolder = fullfile(saveVideoDir,['image_',num2str(iNeighbor)]);
            checkDir(inputCropFolder)
            inputCropFolders{iNeighbor1i} = inputCropFolder;
        end
        
        % for all frames
        for iFrame = 1:nFrames
            frameName = frameNames{iFrame};
            
            % argument
            gt = imread(fullfile( ...
                gtDir, ...
                videoName, ...
                'GT', ...
                frameName));
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
%                     window = [hStart,wStart,cropWidth-1,cropWidth-1];
                    gtCrop = gt(isHCrop,isWCrop,:);
                        
                    [~,name,ext] = fileparts(frameName);
                    cropName = sprintf('%s_%02d%s', ...
                        name,(iArgument-1)*nCrops+iCrop,ext);
                    gtCropDir = [gtCropFolder,'/',cropName];
%                     disp(gtCropDir);
%                     imwrite(gtCrop,gtCropDir);
                    for iNeighbor1i = 1:widthNeighbor
                        input = inputsArg{iNeighbor1i}{iArgument};
                        inputCrop = input(isHCrop,isWCrop,:);
                        
                        inputCropDir = [inputCropFolders{iNeighbor1i},'/',cropName];
%                         disp(inputCropDir);
%                         imwrite(inputCrop,inputCropDir);
                    end
                end
            end
            % crop - end
            qUpdateWaitbar.send(1/nFrames/nVideos);
%             if iFrame == 10
%                 break;
%             end
        end
%         break;
    end
end
%% Clean
delete(processBar);
delete(lUpdateWaitbar);

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

function updateWaitbar(progress)
global nDone nAlignments nVideos processBar;
% disp([nDone nAlignments nVideos])
nDone = nDone+progress;
x = nDone/nAlignments/nVideos;
waitbar(x,processBar,sprintf('Generating... %.2f%%, %.2f minutes left.',x*100,toc/x*(1-x)/60));
end
