classdef Weights
    properties
        model;
        w1;  w2;  w3;
        b1;  b2;  b3;
        c1p; c2p; c3p;
        l1Maps;
        l2Maps;
        l3Maps;
    end
    methods
        function obj = Weights(model)
            obj.model = model;
            
            %% load CNN model parameters
            load(model);
            
            [p2, obj.l1Maps   ] = size(weights_conv1);  obj.c1p = sqrt(p2);
            [~, p2, obj.l2Maps] = size(weights_conv2);  obj.c2p = sqrt(p2);
            [obj.l3Maps, p2   ] = size(weights_conv3);  obj.c3p = sqrt(p2);
            

            % Reshape weights
            obj.w1 = reshape(weights_conv1 , obj.c1p, obj.c1p, obj.l1Maps);
            obj.w2 = weights_conv2;
            obj.w3 = reshape(weights_conv3', obj.c3p, obj.c3p, obj.l3Maps);

            
        end
        
    end
end