function im_h = SRCNN(model, im_b)

%% load CNN model parameters
load(model);
[conv1_patchsize2,conv1_filters] = size(weights_conv1);
conv1_patchsize = sqrt(conv1_patchsize2);
[conv2_channels,conv2_patchsize2,conv2_filters] = size(weights_conv2);
conv2_patchsize = sqrt(conv2_patchsize2);
[conv3_channels,conv3_patchsize2] = size(weights_conv3);
conv3_patchsize = sqrt(conv3_patchsize2);
[hei, wid] = size(im_b);

weights_conv1_h = weights_conv1;
weights_conv2_h = weights_conv2;
weights_conv3_h = weights_conv3;

% Reshape weights
weights_conv1 = reshape(weights_conv1 , conv1_patchsize, conv1_patchsize, conv1_filters);

newTensor = NaN(conv2_channels, conv2_filters, conv2_patchsize, conv2_patchsize);
for i = 1 : conv2_filters
    for j = 1 : conv2_channels
        subfilter = reshape(weights_conv2(j,:,i), conv2_patchsize, conv2_patchsize);
        newTensor(j,i,:,:) = subfilter;
    end
end
weights_conv2 = newTensor;

weights_conv3 = reshape(weights_conv3', conv3_patchsize, conv3_patchsize, conv3_channels);


%% conv1
conv1_data = zeros(hei, wid, conv1_filters);
for i = 1 : conv1_filters
    
    
    subfilter = weights_conv1(:,:,i);
    subdata   = im_b;
    conved    = imfilter(subdata, subfilter, 'same', 'replicate');
    
    % CONV
    conv1_data(:,:,i) = conved;
    
    % RELU
    conv1_data(:,:,i) = max(conv1_data(:,:,i) + biases_conv1(i), 0);
end

%% conv2
conv2_data = zeros(hei, wid, conv2_filters);
for i = 1 : conv2_filters
    
    % CONV
    for j = 1 : conv2_channels
        subfilter = reshape(weights_conv2(j,i,:,:), conv2_patchsize, conv2_patchsize);
        subdata   = conv1_data(:,:,j);
        conved    = imfilter(subdata, subfilter, 'same', 'replicate');
        
        conv2_data(:,:,i) = conv2_data(:,:,i) + conved;
    end
    
    % RELU
    conv2_data(:,:,i) = max(conv2_data(:,:,i) + biases_conv2(i), 0);
end

%% conv3
conv3_data = zeros(hei, wid);
for i = 1 : conv3_channels
    
    subfilter = weights_conv3(:,:,i);
    subdata   = conv2_data(:,:,i);
    conved    = imfilter(subdata, subfilter, 'same', 'replicate');
    
    % CONV
    conv3_data(:,:) = conv3_data(:,:) + conved;
end

% SRCNN reconstruction

% RELU
im_h = conv3_data(:,:) + biases_conv3;