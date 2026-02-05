classdef dataSplitLayer < nnet.layer.Layer...  %#codegen
        & nnet.layer.Formattable ...
        & nnet.layer.Acceleratable
    properties
        spaceSelect=1;
    end

    methods
        function layer = dataSplitLayer(spaceSelect,name)
            layer.spaceSelect=spaceSelect;
            layer.Name = name;
        end

        function Z = predict(layer, X)
            Z = X(:,layer.spaceSelect,:,:);
        end
    end

end