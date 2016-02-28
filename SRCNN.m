function im_h = SRCNN(model, im_b)

%% load CNN model parameters
load(model);
[p2,l1Maps] = size(weights_conv1);                c1p = sqrt(p2);
[conv2_channels,p2,l2Maps] = size(weights_conv2); c2p = sqrt(p2);
[l3Maps,p2] = size(weights_conv3);                c3p = sqrt(p2);

weights_conv1_h = weights_conv1;
weights_conv2_h = weights_conv2;
weights_conv3_h = weights_conv3;

% Reshape weights
weights_conv1 = reshape(weights_conv1 , c1p, c1p, l1Maps);
weights_conv3 = reshape(weights_conv3', c3p, c3p, l3Maps);


%%

[hei, wid] = size(im_b);

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
        subfilter = reshape(weights_conv2(j,:,i), c2p, c2p);
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