classdef reshapeLayer < nnet.layer.Layer...  %#codegen
        & nnet.layer.Formattable ...
        & nnet.layer.Acceleratable

    properties
        permuteOrder (1,:) double
        channelOrder string
    end

    methods
        function layer = reshapeLayer(name,permuteOrder,channelOrder)
            layer.Name = name;
            layer.permuteOrder=permuteOrder;
            layer.channelOrder=channelOrder;
        end

        function Z = predict(layer, X)
                Z= dlarray(permute(stripdims(X),layer.permuteOrder),layer.channelOrder);
        end
    end

end