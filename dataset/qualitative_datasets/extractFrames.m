% Clean
clear
close all

%% Read Videos
videoExt = '.mp4';
list = dir(['*',videoExt]);
videoNames = {list.name};

for videoName = videoNames
    videoName = videoName{1};
    message = ['Reading video ',videoName,'...'];
    disp(message);
    
    video = VideoReader(videoName);
    nFrames = video.NumberOfFrames;
    video = VideoReader(videoName);
    [~,name,~] = fileparts(videoName);
    outFolder = fullfile(name,'input');
    if ~exist(outFolder,'dir')
        mkdir(outFolder)
    end
    
    processBar = waitbar(0,message,'CreateCancelBtn','setappdata(gcbf,''canceling'',1)');
    iFrame = 1;
    while hasFrame(video)
        if getappdata(processBar,'canceling')
            break
        end
        
        frame = readFrame(video);
        imwrite(frame,fullfile(outFolder,[sprintf('%05d',iFrame),'.jpg']));
        iFrame = iFrame+1;
        waitbar(iFrame/nFrames,processBar);
    end
    if getappdata(processBar,'canceling')
        delete(processBar);
        break
    end
    delete(processBar);
end
