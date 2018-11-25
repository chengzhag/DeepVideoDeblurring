for iPatch = 1:size(batchGTTorch,1)
    patchOut = reshape(squeeze((permute(batchGTTorch(iPatch,:,:,:),[1,3,4,2]))),[128,128,3,1]);
    patchIn = reshape(squeeze((permute(batchInputTorch(iPatch,:,:,:),[1,3,4,2]))),[128,128,3,5]);
    patch = zeros(128,128,3,6,'uint8');
    patch(:,:,:,1:5) = patchIn;
    patch(:,:,:,end) = patchOut;
    montage(patch)
    pause(1)
end