function generateAllAlignments
%% Clean
clear;
close all;
warning off

%% Parameters
nFramesMax = 100;
nAlignment = 3;
alignments = {'_nowarp','_OF','_homography'};
datasetDir = '..\dataset\qualitative_datasets';
breakpointFile = 'breakpoint_generateAllAlignments.txt';

%% Read breakpoint
fileID = fopen(breakpointFile,'r');
breakpoint = textscan(fileID,'%s');
breakpoint = unique(breakpoint{1});
fclose(fileID);

%% Scan videos
videoFolders = dir(datasetDir);
maskFolders = [videoFolders.isdir];
videoFolders = videoFolders(maskFolders);
videoFolders = videoFolders(3:end);
videoFolders = {videoFolders.name};

%% Maintain breakpoint history
fileID = fopen(breakpointFile,'w');
breakpoint = intersect(breakpoint,videoFolders);
for i = 1:length(breakpoint)
    fprintf(fileID,'%s\n',breakpoint{i});
end
fclose(fileID);
videoFolders = setdiff(videoFolders,breakpoint);

%% Generation Prepare
nVideos = length(videoFolders);
nDone = 0;
processBar = waitbar(0,'Generating... ');
qUpdateWaitbar = parallel.pool.DataQueue;
lUpdateWaitbar = qUpdateWaitbar.afterEach(@(progress) updateWaitbar(progress));
qWriteBreakpoint = parallel.pool.DataQueue;
fileID = fopen(breakpointFile,'a');
lBreakpoint = qWriteBreakpoint.afterEach(@(f) fprintf(fileID,'%s\n',f));

%% Start Generating
tic
parfor iVideo = 1:nVideos
    warning off
    videoFolder = videoFolders{iVideo};
    fprintf('Generating Alignment of video %s (%d/%d)\n',videoFolder,iVideo,nVideos);
    from = fullfile(datasetDir,videoFolder,'input');
    % circshift to make waitbar more accurate
    for iAlignment = circshift(0:nAlignment-1,mod(iVideo,nAlignment))
        alignment = alignments{iAlignment+1};
        to = fullfile(sprintf('../data/testing_real_all_nostab%s',alignment),videoFolder);
        alignVideo(iAlignment,from,to,nFramesMax,qUpdateWaitbar);
    end
    disp([videoFolder,' done!']);
    qWriteBreakpoint.send(videoFolder);
end

%% Clean
delete(processBar);
delete(lUpdateWaitbar);
delete(lBreakpoint);
fclose(fileID);

    function updateWaitbar(progress)
        nDone = nDone+progress;
        x = nDone/nAlignment/nVideos;
        waitbar(x,processBar,sprintf('Generating... %.2f%%, %.2f minutes left.',x*100,toc/x*(1-x)/60));
    end
end
