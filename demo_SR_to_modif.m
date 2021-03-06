format compact
close all;
clear;

%% read ground truth image
im  = imread('Set5/butterfly_GT.bmp');
%im  = imread('Set14/zebra.bmp');

%% set parameters
up_scale = 3;
model = 'model/9-5-5(ImageNet)/x3.mat';
% up_scale = 3;
% model = 'model\9-3-5(ImageNet)\x3.mat';
% up_scale = 3;
% model = 'model\9-1-5(91 images)\x3.mat';
% up_scale = 2;
% model = 'model\9-5-5(ImageNet)\x2.mat'; 
% up_scale = 4;
% model = 'model\9-5-5(ImageNet)\x4.mat';

%% work on illuminance only
if size(im,3)>1
    im = rgb2ycbcr(im);
    im = im(:, :, 1);
end
im_gnd = modcrop(im, up_scale);
im_gnd = single(im_gnd)/255;

params = Weights(model);

%% bicubic interpolation
im_l = imresize(im_gnd, 1/up_scale, 'bicubic'); % Low resolution
im_b = imresize(im_l  ,   up_scale, 'bicubic'); % Bicubic high res

%% SRCNN
im_h = SRCNN(model, im_b);  % SRCNN high res

%% remove border
im_h   = shave(uint8(im_h   * 255), [up_scale, up_scale]);
im_gnd = shave(uint8(im_gnd * 255), [up_scale, up_scale]);
im_b   = shave(uint8(im_b   * 255), [up_scale, up_scale]);

%% compute PSNR
psnr_bic   = compute_psnr(im_gnd, im_b);
psnr_srcnn = compute_psnr(im_gnd, im_h);

%% show results
fprintf('PSNR for Bicubic Interpolation: %f dB\n', psnr_bic  );
fprintf('PSNR for SRCNN Reconstruction : %f dB\n', psnr_srcnn);

%figure, imshow(im_b); title('Bicubic Interpolation');
%figure, imshow(im_h); title('SRCNN Reconstruction');

%imwrite(im_b, ['Bicubic Interpolation' '.bmp']);
%imwrite(im_h, ['SRCNN Reconstruction' '.bmp']);


% Console output:
% PSNR for Bicubic Interpolation: 24.037923 dB
% PSNR for SRCNN Reconstruction:  27.953003 dB
%