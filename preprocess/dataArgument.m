
%% Clean
clear
close all

%% Parameters
global nAlignments nVideos;
alignments = {'_nowarp'};%,'_OF','_homography'};
nAlignments = length(alignments);
inputDir = '../data';
gtDir = '../dataset/quantitative_datasets';
inputFolderPrefix = 'training_real_all_nostab';
saveDir = '../data';
saveFolderPrefix = 'training_augumented_uncroped_all_nostab';
frameExt = '.jpg';
argumentZoom = {1/4, 1/3, 1/2, 1};
nFramesAll = 6708;

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

%% Argument
% for all alignments
tic;
for iAlignment = 1:nAlignments
    alignment = alignments{iAlignment};
    nVideos = length(videoNames{iAlignment});
    saveAlignFolder = fullfile(saveDir,[saveFolderPrefix,alignment]);
    checkDir(saveAlignFolder);
    % for all videos
    nFramesDone = 0;
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
            input = imread(fullfile( ...
                inputFolders{iAlignment}, ...
                videoName, ...
                'image_0', ...
                frameName));
            inputsArg = argument(input,argumentZoom);
            % argument - end
            
            pushArg(gtsArg,inputsArg,saveVideoDir,frameName);
            nFramesDone = nFramesDone+1;
            ms = toc*(nFramesAll - nFramesDone)/60;
            tic;
            hours = floor(ms/60);
            mins = mod(ms,60);
            fprintf('Generating... %.2f%%, %d hours %.1f minutes left.\n', ...
                nFramesDone/nFramesAll*100,hours,mins);
        end
        % clear patch index -- start a new batch
        iPatch = 1;
    end
end
%% Functions
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


function pushArg(gtsArg,inputsArg,saveFolder,frameName)
[~,name,~] = fileparts(frameName);
GTfolder = fullfile(saveFolder,'GT');
inputFolder = fullfile(saveFolder,'input');
checkDir(GTfolder);
checkDir(inputFolder);
% disp(GTfolder);
% disp(inputFolder);
for iArg = 1:length(gtsArg)
%     montage({gtsArg{iArg},inputsArg{iArg}});
%     disp(size(gtsArg{iArg}))
%     pause(1);
    saveName = sprintf('%s_%02d.jpg',name,iArg-1);
    GTdir = fullfile(GTfolder,saveName);
    inputDir = fullfile(inputFolder,saveName);
%     disp(GTdir);
%     disp(inputDir);
    imwrite(gtsArg{iArg},GTdir);
    imwrite(inputsArg{iArg},inputDir);
end
end
