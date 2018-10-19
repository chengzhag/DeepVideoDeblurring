function alignVideo(alignmentType,from,to,nFramesMax,qUpdateWaitbar)
if nargin == 4
    qUpdateWaitbar = [];
end

% alignmentType: 0 for nowarp, 1 for optical flow, 2 for homography, 3 for similarity
addpath(genpath('thirdparty'));

%% Parameters
frames = dir(fullfile(from,'*.jpg'));
if isempty(frames)
    return
end
for l = -2:2
    checkDir(fullfile(to,['image_',num2str(l)]));
end

%% Align
nFrames = min(length(frames),nFramesMax);
frameDirs = @(iFrame) fullfile(from,frames(iFrame).name);
fr_cnt = 0;
for iFrame = 1:min(nFrames,nFramesMax)
    fr_cnt = fr_cnt+1;
    % save image_1 to image_5
    v0 = im2double(imread(frameDirs(iFrame)));
    v0g = single(rgb2gray(v0));
    [h,w,~] = size(v0);
    
    for l = -2:2
        if l ~= 0
            vi = im2double(imread(frameDirs(max(min(iFrame+l,length(frames)),1))));
            vig = single(rgb2gray(vi));
            if alignmentType == 0
                v_i0 = vi;
            elseif alignmentType == 1
                flo_i0 = genFlow(v0g, vig);
                [v_i0, ~] = warpToRef(v0, vi, flo_i0);
            elseif alignmentType == 2
                v_i0 = homographyAlignment(v0,vi,0);
            elseif alignmentType == 3
                v_i0 = similarityAlignment(v0,vi,0);
            end
        else
            v_i0 = v0;
        end
        imwrite(v_i0, fullfile(to,['image_',num2str(l)],[sprintf('%05d',iFrame),'.jpg']));
    end
    if isa(qUpdateWaitbar,'parallel.pool.DataQueue')
        qUpdateWaitbar.send(1/nFrames);
    end
    
end



