function export_network_as_TF(network,pythonPackageName)
%GET_NETWORK_INFO   display the structure and input size
%   of a deep learning network.
%   author: Zhaozheng Hou (George)
%
% inputSize = get_network_info(network)
% parameter:
%   - net: the network to analysis
% output:
%   - inputSize: the input size of the network

%exportNetworkToTensorFlow  Export a MATLAB deep learning network to
%TensorFlow.
%
% exportNetworkToTensorFlow(network, pythonPackageName) exports a MATLAB
% deep learning network to a TensorFlow model in a custom Python package
% <pythonPackageName>. The input network can be a SeriesNetwork,
% DAGNetwork, dlnetwork, layerGraph, or layer array. pythonPackageName is a
% string specifying the name of the custom Python package that the function
% creates. The example below shows how to export a network from MATLAB and
% import it into TensorFlow.
%
% Example:
%     % Step 1. Export a MATLAB pretrained network into a Python package
%     % named "myModel":
%     net = squeezenet;
%     exportNetworkToTensorFlow(net,"myModel")
%
%     # Step 2. In Python, import the package, and then load the model by
%     # calling the package's load_model() function:
%     import myModel
%     model = myModel.load_model()
%
%     # Step 3 (optional). In Python, save the model in SavedModel format:
%     model.save("mySavedModel.keras")

% Copyright 2022-2023 The Mathworks, Inc.

%% validate
if (isa(network,"dlnetwork"))
    validateattributes(network,{'dlnetwork'},{'scalar'});
else
    validateattributes(network,{'string','char'},{'scalartext'});
    network=load("core\"+string(network)+".mat",string(network));
    network=struct2cell(network);
    network=network{1};
end
net_input=network.InputNames;
validateattributes(net_input,{'cell'},{'scalar'});
n=network.getLayer(network.InputNames{1}).InputSize;

pythonPackageName=string(pythonPackageName);

if (~network.Initialized)
    warning("The given network is not yet initialized.");
end

%% process
if (mod(n(2),10)==0) % is a generator

    network=network.addLayers(flattenLayer(Name='flatten'));
    network=network.disconnectLayers('gapool_2','multiplication/in1');
    network=network.connectLayers('gapool_2','flatten');
    network=network.removeLayers({'concat','formattingLayer_1_1','formattingLayerasdf'});
    network=network.connectLayers('flatten','multiplication/in1');
    network=network.connectLayers('GeneLevelOutput','multiplication/in2');
    network=network.initialize();

else% is a discriminator
    n=n(2)-1;

    FirstData = convolution2dLayer([1,n+1],1,"Name","FirstData");
    FirstData.WeightLearnRateFactor=0;
    FirstData.Weights=[1,zeros(1,n)];
    FirstData.Bias=0;
    FirstData.BiasLearnRateFactor=0;

    RefData = convolution2dLayer([1,n+1],n,"Name","RefData");
    RefData.WeightLearnRateFactor=0;
    RefData.Weights=permute([zeros(n,1),eye(n)],[3,2,4,1]);
    RefData.Bias=zeros(1,1,n);
    RefData.BiasLearnRateFactor=0;

    tempLayers = [
        reshapeLayer("formattingLayer_all",[3 2 1 4],"SSCB");
        RefData;
        reshapeLayer("formattingLayer_ref",[3 2 1 4],"SSCB")];
    network=network.replaceLayer('split_reference',tempLayers);
    tempLayers=[
        FirstData;
        reshapeLayer("formattingLayer_first",[3 2 1 4],"SSCB")];
    network=network.replaceLayer('split_first',tempLayers);
    network=network.disconnectLayers('batchnorm_1','FirstData');
    network=network.connectLayers('formattingLayer_all','FirstData');
    network=network.initialize();
end

warnStruct = warning('off',...
    'nnet_cnn_kerasimporter:keras_importer:exporterConverterForUnsupportedLayer');
exportNetworkToTensorFlow(network,pythonPackageName);
warning(warnStruct);

temp=readlines(pythonPackageName+"\model.py");
temp(8)="# "+temp(8);
temp=strrep(temp,"reshapeLayer()","layers.Permute((3,2,1))");
writelines(temp,pythonPackageName+"\model.py","WriteMode","overwrite");
delete(pythonPackageName+"\customLayers\reshapeLayer.py");
rmdir(pythonPackageName+"\customLayers\");
end