
%% Clean
clear
close all

%% Parameters
date = '1220';
alignments = {'_nowarp','_homography','_OF'};%{'_nowarp','_homography','_OF'};
outputDir = 'E:\projects\DeepVideoDeblurring\outImg';
outputFolderPrefix = [date '_validating_model2_symskip_nngraph2_deeper'];
gtDir = 'E:\projects\DeepVideoDeblurring\dataset\quantitative_datasets';
global frameExt
frameExt = '.jpg';

%% Scan videos
outputFolders = {};
for alignment = alignments
    outputFolders{end+1} = [outputFolderPrefix,alignment{1}];
end
outputFolders = fullfile(outputDir,outputFolders);
% videoNames = {};
% for outputFolder = outputFolders
%     videoNames{end+1} = dir(outputFolder{1});
%     maskFolders = [videoNames{end}.isdir];
%     videoNames{end} = videoNames{end}(maskFolders);
%     videoNames{end} = videoNames{end}(3:end);
%     videoNames{end} = {videoNames{end}.name};
% end

validset = {'IMG_0030' 'IMG_0049' 'IMG_0021' '720p_240fps_2' 'IMG_0032' ...
'IMG_0033' 'IMG_0031' 'IMG_0003' 'IMG_0039' 'IMG_0037'};% from git rep DeepVideoDeburring

%% Evaluate PSNR
align2PSNRs = zeros(length(alignments),length(validset)+1);
for iAlignment = 1:length(alignments)
    alignment = alignments{iAlignment};
    
    %% scan validset frames
    frameDirs = scanFrames(outputFolders{iAlignment},gtDir,validset);
    
    %% evaluate PSNR
    for iVideo = 1:size(frameDirs,1)
        avgPSNR = 0;
        nFrame = size(frameDirs,2);
        parfor iFrame = 1:nFrame
            gtFrameDir = frameDirs{iVideo,iFrame,1};
            outputFrameDir = frameDirs{iVideo,iFrame,2};
            gt = imread(gtFrameDir);
            output = imread(outputFrameDir);
            scores = evaluate_align(gt,output,'PSNR');
            avgPSNR = avgPSNR+scores.PSNR;
%             scores = metrix_mux(gt,output,'PSNR');
%             avgPSNR = avgPSNR+scores;
        end
        avgPSNR = avgPSNR/nFrame;
        align2PSNRs(iAlignment,iVideo) = avgPSNR;
        fprintf('video %d/%d done!\n',iVideo,size(frameDirs,1))
    end
    
end
align2PSNRs(:,end) = mean(align2PSNRs(:,1:length(validset)),2)

%% functions 
function frameDirs = scanFrames(outputFolder,gtDir,videoNames)
global frameExt
% for all videos
disp('Scanning frames...');
frameDirs = {};
iFrameAll = 1;
for iVideo = 1:length(videoNames)
    videoName = videoNames{iVideo};
    outputFrameFolder = fullfile(outputFolder,videoName);
    outputFrameNames = dir(fullfile(outputFrameFolder,['*',frameExt]));
    outputFrameNames = {outputFrameNames.name};
    gtFrameFolder = fullfile(gtDir,videoName,'GT');
    gtFrameNames = dir(fullfile(gtFrameFolder,['*',frameExt]));
    gtFrameNames = {gtFrameNames.name};
    for iFrame = 1:length(outputFrameNames)
        frameDirs{iVideo,iFrame,1} = [gtFrameFolder,'/',gtFrameNames{iFrame}];
        frameDirs{iVideo,iFrame,2} = [outputFrameFolder,'/',outputFrameNames{iFrame}];
        iFrameAll = iFrameAll+1;
    end
    fprintf('Found %d frames, video %d/%d\n',iFrameAll-1,iVideo,length(videoNames))
end
fprintf('Found %d frames\n',iFrameAll-1)
end