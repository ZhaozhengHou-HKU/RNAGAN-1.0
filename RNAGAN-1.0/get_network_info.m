function inputSize = get_network_info(network)
%GET_NETWORK_INFO   display the structure and input size
%   of a deep learning network.
%   author: Zhaozheng Hou (George)
%   
% inputSize = get_network_info(network)
% parameter:
%   - net: the network to analysis
% output:
%   - inputSize: the input size of the network

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

if (~network.Initialized)
    warning("The given network is not yet initialized.");
end

%% process
inputSize=network.getLayer(net_input{1}).InputSize;

plot(network);
end