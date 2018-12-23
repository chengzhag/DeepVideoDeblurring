function scores = evaluate_align(groundTruth,deblurred,metric)
% BENCHMARK_EVAL_IMAGE computes various quality measures given
% a deblurred image and the corresponding ID, i.e. you have to
% provide the image and kernel number
%
% Michael Hirsch and Rolf Koehler (c) 2012

addpath(genpath('image_quality_algorithms'))

% -----------------------------------------------------------------
% Check that input images are 8bit.
% To use the image quality assessment algorithms the images have to be in 8
% bit, with a maximum value of 255, so not normalized.
% -----------------------------------------------------------------
if ( ~isa(deblurred, 'uint8') || size(deblurred,3)~=3 )
    error('Input image has to be a color image of type uint8!');
end

% z will denote the estimated, i.e. deblurred image
z = double(deblurred);

% Create constant image which will act as a mask
c = ones(size(deblurred));

% Initialize containers for quality measures
mse   = [];
mssim = [];
vif   = [];
ifc   = [];
psnr  = [];
mad   = [];


% =================================================================
% compute the image quality measure score
% -----------------------------------------------------------------


% Load ground truth image, denoted by x
x  = double(groundTruth);
xf = fft2(mean(x,3));

% Scale image intensity
zs = (sum(vec(x.*z)) / sum(vec(z.*z))) .* z;
zf = fft2(mean(zs,3));

% Estimate shift by registering deblurred to ground truth image
[output Greg] = dftregistration(xf, zf, 1);
shift = output(3:4);

% Apply shift
cr = imshift(double(c), shift, 'same');
zr = imshift(double(zs), shift, 'same');
xr = x.*cr;

% Compute various quality measures
switch metric
    case 'MSE'
        mse   = [mse   metrix_mux(xr, zr, 'MSE')];
    case 'MSSIM'
        mssim = [mssim metrix_mux(xr, zr, 'MSSIM')];
    case 'VIF'
        vif   = [vif   metrix_mux(xr, zr, 'VIF')];
    case 'IFC'
        ifc   = [ifc   metrix_mux(xr, zr, 'IFC')];
    case 'PSNR'
        psnr  = [psnr  metrix_mux(xr, zr, 'PSNR')];
    case 'MAD'
        MAD_temp = MAD_index(xr, zr);
        mad   = [mad   MAD_temp.MAD];
end


% =================================================================
% Save results
% -----------------------------------------------------------------
switch metric
    case 'MSE'
        scores.MSE   = mse;
    case 'MSSIM'
        scores.MSSIM = mssim;
    case 'VIF'
        scores.VIF   = vif;
    case 'IFC'
        scores.IFC   = ifc;
    case 'PSNR'
        scores.PSNR  = psnr;
    case 'MAD'
        scores.MAD   = mad;
end

return

