function im_h = SRCNN(model, im_b)

%% load CNN model parameters
load(model);
[conv1_patchsize2,l1Maps] = size(weights_conv1);
conv1_patchsize = sqrt(conv1_patchsize2);
[conv2_channels,conv2_patchsize2,l2Maps] = size(weights_conv2);
conv2_patchsize = sqrt(conv2_patchsize2);
[l3Maps,conv3_patchsize2] = size(weights_conv3);
conv3_patchsize = sqrt(conv3_patchsize2);
[hei, wid] = size(im_b);


c1InMaps = 1;
c1OtMaps = l1Maps;

c2InMaps = l1Maps;
c2OtMaps = l2Maps;

c3InMaps = l1Maps;
c3OtMaps = 1;
weights_conv1_h = weights_conv1;
weights_conv2_h = weights_conv2;
weights_conv3_h = weights_conv3;

% Reshape weights
weights_conv1 = reshape(weights_conv1 , conv1_patchsize, conv1_patchsize, l1Maps);

newTensor = NaN(l1Maps, l2Maps, conv2_patchsize, conv2_patchsize);
for i = 1 : l2Maps
    for j = 1 : l1Maps
        subfilter = reshape(weights_conv2(j,:,i), conv2_patchsize, conv2_patchsize);
        newTensor(j,i,:,:) = subfilter;
    end
end
weights_conv2 = newTensor;

weights_conv3 = reshape(weights_conv3', conv3_patchsize, conv3_patchsize, l3Maps);


%% conv1
conv1_data = zeros(hei, wid, l1Maps);
for i = 1 : l1Maps
    
    subfilter = weights_conv1(:,:,i);
    subdata   = im_b;
    conved    = imfilter(subdata, subfilter, 'same', 'replicate');
    
    bias      = biases_conv1(i);
    
    conv1_data(:,:,i) = max(conved + bias, 0);
end

%% conv2
conv2_data = zeros(hei, wid, l2Maps);
for i = 1 : l2Maps
    
    % CONV
    for j = 1 : l1Maps
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
for i = 1 : l3Maps
    
    subfilter = weights_conv3(:,:,i);
    subdata   = conv2_data(:,:,i);
    conved    = imfilter(subdata, subfilter, 'same', 'replicate');
    
    % CONV
    conv3_data(:,:) = conv3_data(:,:) + conved;
end

% SRCNN reconstruction

% RELU
im_h = conv3_data(:,:) + biases_conv3;