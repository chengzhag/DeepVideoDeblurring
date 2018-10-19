function qualitativelyCompare
% Clean
clear
close all

%% Read dir
nFramesMax = 100;
aligns = {'Input','DBN+Flow','DBN+Homog','DBN+Noalign'};
nAlignment = length(aligns);

% root = '..\results\DeepVideoDeblurring_Results_Videos_Only';
% saveImg2 = 'original_failure';
% fileExt = '.mp4';
% list = dir(fullfile(root,aligns{1},['*',fileExt]));
% videoNames = {list.name};

videoFolders = {'..\dataset\qualitative_datasets' ...
    '..\outImg\1018_model2_symskip_nngraph2_deeper_OF_real' ...
    '..\outImg\1018_model2_symskip_nngraph2_deeper_homography_real' ...
    '..\outImg\1018_model2_symskip_nngraph2_deeper_nowarp_real'};
saveImg2 = 'failure';
list = dir(videoFolders{2});
list = list([list.isdir]);
list = list(3:end);
videoNames = {list.name};
fileExt = '.jpg';

nVideos = length(videoNames);

%% Read videos
nRow = floor(sqrt(length(aligns)));
nCol = ceil(length(aligns)/nRow);
doSkipAsk = false;
while 1
    if ~doSkipAsk
        iVideo = askWhichVideo;
    end
    videoName = videoNames{iVideo};
    
    % Prepare reading
    if strcmp(fileExt,'.mp4')
        videoFrames = cell(1,length(aligns));
        video = VideoReader(fullfile(root,aligns{1},videoName));
        nFrames = video.NumberOfFrames;
        height = video.Height;width = video.Width;
    elseif strcmp(fileExt,'.jpg')
        inputFrameFolder = fullfile(videoFolders{1},videoName,'input');
        frameList = dir(fullfile(inputFrameFolder,['*',fileExt]));
        frameNames = {frameList.name};
        nFrames = length(frameNames);
        imSample = imread(fullfile(inputFrameFolder,frameNames{1}));
        [height,width,~] = size(imSample);
    end
    nFrames = min(nFrames,nFramesMax);
    message = ['Reading video ',videoName,'...'];
    disp(message);
    processBar = waitbar(0,message);
    nDone = 0;
    qUpdateWaitbar = parallel.pool.DataQueue;
    lUpdateWaitbar = qUpdateWaitbar.afterEach(@(progress) updateWaitbar(progress));
    parfor iAlign = 1:length(aligns)
        message = ['Reading video ',aligns{iAlign},'\\',videoName,'...'];
        disp(message);
        % Allocate memory
        vf = zeros(height,width,3,nFrames,'uint8');
        % Read video
        if strcmp(fileExt,'.mp4')
            video = VideoReader(fullfile(root,aligns{iAlign},videoName));
            for iFrame = 1:nFrames
                if hasFrame(video)
                    vf(:,:,:,iFrame) = readFrame(video);
                end
                qUpdateWaitbar.send(1/nFrames);
            end
        elseif strcmp(fileExt,'.jpg')
            imageFolder = videoFolders{iAlign};
            if iAlign == 1
                frameFolder = fullfile(imageFolder,videoName,'input');
            else
                frameFolder = fullfile(imageFolder,videoName);
            end
            
            for iFrame = 1:nFrames
                frameDir = fullfile(frameFolder,frameNames{iFrame});
                if exist(frameDir,'file')
                    vf(:,:,:,iFrame) = imread(frameDir);
                end
                qUpdateWaitbar.send(1/nFrames);
            end
        end
        videoFrames{iAlign} = vf;
    end
    delete(processBar)
    
    disp(['Viewing video ',videoName,'...']);
    iFrame = 1;
    flagPlay = false;
    frameDelay = 0.1;
    close all;
    fi = figure('Name',videoName);
    while 1
        montageFrames = cell(1,length(aligns));
        for iAlign = 1:length(aligns)
            montageFrames{iAlign} = videoFrames{iAlign}(:,:,:,iFrame);
        end
        figure(fi);
        montage(montageFrames);
        title([videoName,' Frame ',num2str(iFrame),'/',num2str(nFrames)],'FontSize',24);
        
        if flagPlay
            pause(0.1);
            iFrame = iFrame+1;
            k = 1;
        else
            k = waitforbuttonpress;
        end
        if k
            key = uint8(get(gcf,'CurrentCharacter'));
            if key == 'q'
                close all;
                doSkipAsk = false;
                break;
            elseif key == 31
                iFrame = iFrame+1;
            elseif key == 30
                iFrame = iFrame-1;
            elseif key == 28
                iFrame = iFrame-5;
            elseif key == 29
                iFrame = iFrame+5;
            elseif key == 32
                flagPlay = true;
            elseif key == 's'
                flagPlay = false;
            elseif key == 13
                set(gcf,'outerposition',get(0,'screensize'));
                framwWrite = getframe(gcf);
                framwWrite = frame2im(framwWrite);
                [~,name,~] = fileparts(videoName);
                imwrite(framwWrite,fullfile(saveImg2,[name,'_',num2str(iFrame),'.png']));
            elseif key == 61
                iVideo = iVideo+1;
            elseif key == 45
                iVideo = iVideo-1;
            end
            % limit Params
            if iFrame > nFrames
                iFrame = 1;
            end
            if iFrame < 1
                iFrame = nFrames;
            end
            if key == 61 || key == 45
                if iVideo > nVideos
                    iVideo = 1;
                end
                if iVideo < 1
                    iVideo = nVideos;
                end
                doSkipAsk = true;
                break;
            end
        end
    end
end
    function iVideo = askWhichVideo
        disp('Choose a video to read:');
        for iVideo = 1:nVideos
            disp([num2str(iVideo),'. ',videoNames{iVideo}]);
        end
        iVideo = input('Select a number:');
    end
    function updateWaitbar(progress)
        nDone = nDone+progress;
        x = nDone/nAlignment;
        waitbar(x,processBar,sprintf('Reading... %.2f%%',x*100));
    end
end