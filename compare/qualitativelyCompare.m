% Clean
clear
close all

%% Read dir
root = '..\results\DeepVideoDeblurring_Results_Videos_Only';
saveImg2 = 'original_failure';
aligns = {'Input','DBN+Flow','DBN+Homog','DBN+Noalign'};
videoExt = '.mp4';

list = dir(fullfile(root,aligns{1},['*',videoExt]));
videoNames = {list.name};

%% Read videos
nRow = floor(sqrt(length(aligns)));
nCol = ceil(length(aligns)/nRow);
while 1
    disp('Choose a video to read:');
    for iVideo = 1:length(videoNames)
        disp([num2str(iVideo),'. ',videoNames{iVideo}]);
    end
    iVideo = input('Select a number:');
    
    disp(['Reading video ',videoNames{iVideo},'...']);
    videoFrames = cell(1,length(aligns));
    nFrames = 1e6;
    for iAlign = 1:length(aligns)
        message = ['Reading video ',aligns{iAlign},'\\',videoNames{iVideo},'...'];
        disp(message);
        processBar = waitbar(0,message);
        % Allocate memory
        video = VideoReader(fullfile(root,aligns{iAlign},videoNames{iVideo}));
        nFrames = min(nFrames,video.NumberOfFrames);
        vf = zeros(video.Height,video.Width,3,video.NumberOfFrames,'uint8');
        % Read video
        video = VideoReader(fullfile(root,aligns{iAlign},videoNames{iVideo}));
        iFrame = 1;
        while hasFrame(video)
            vf(:,:,:,iFrame) = readFrame(video);
            iFrame = iFrame+1;
            waitbar(iFrame/nFrames,processBar);
        end 
        videoFrames{iAlign} = vf;
        delete(processBar)
    end
    
    disp(['Viewing video ',videoNames{iVideo},'...']);
    iFrame = 1;
    flagPlay = false;
    frameDelay = 0.1;
    while 1
        montageFrames = cell(1,length(aligns));
        for iAlign = 1:length(aligns)
            montageFrames{iAlign} = videoFrames{iAlign}(:,:,:,iFrame);
        end
        
        montage(montageFrames);
        title([videoNames{iVideo},' Frame ',num2str(iFrame),'/',num2str(nFrames)],'FontSize',24);
        
        if flagPlay
            pause(0.1);
            iFrame = iFrame+1;
        else
            waitforbuttonpress;
        end
        key = uint8(get(gcf,'CurrentCharacter'));
        if key == 'q'
            close all;
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
            framwWrite = getframe(gcf);
            framwWrite = frame2im(framwWrite);
            [~,name,~] = fileparts(videoNames{iVideo});
            imwrite(framwWrite,fullfile(saveImg2,[name,'_',num2str(iFrame),'.png']));
        end
        if iFrame > nFrames
            iFrame = nFrames;
        end
        if iFrame < 1
            iFrame = 1;
        end
    end
end