function netD = DiscriminatorNetwork(pathwayMatrix,numRef)
%provide pathway in [pathway,genes]
%learnable pathways with all true
%no pathway with an empty matrix with 0 rows


numGene=size(pathwayMatrix,2);

numPathways=size(pathwayMatrix,1);
leakyReluScale=0.01;
featureSize=16;

lgraph = layerGraph();

tempLayers = imageInputLayer([numGene,(numRef+1),1],"Name","Input","Normalization","none");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution2dLayer([numGene 1],numPathways,"Name","Pathways");
if (~all(pathwayMatrix,"all")) % load pathway info
    tempLayers.WeightLearnRateFactor=0;
    tempLayers.Weights=single(permute(pathwayMatrix,[2,3,4,1]));
end
tempLayers.Bias=single(zeros(1,1,numPathways));
tempLayers.BiasLearnRateFactor=0;
lgraph = addLayers(lgraph,tempLayers);

tempLayers = reshapeLayer("formattingLayer_1",[3 2 1 4],"SSCB");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","depthcat")
    leakyReluLayer(leakyReluScale,"Name","leakyrelu_1")
    batchNormalizationLayer("Name","batchnorm_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = dataSplitLayer(1,"split_first");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = dataSplitLayer(2:(numRef+1),"split_reference");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = globalMaxPooling2dLayer("Name","gmpool");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = globalAveragePooling2dLayer("Name","gapool");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    concatenationLayer(2,3,"Name","concat")
    reshapeLayer("formattingLayer_2",[3 2 1 4],"SSCB")
    fullyConnectedLayer(16*featureSize,"Name","fc_1")
    leakyReluLayer(leakyReluScale,"Name","leakyrelu_2")
    batchNormalizationLayer("Name","batchnorm_2")
    dropoutLayer(0.5,"Name","dropout")
    fullyConnectedLayer(4*featureSize,"Name","fc_2")
    leakyReluLayer(leakyReluScale,"Name","leakyrelu_3")
    batchNormalizationLayer("Name","batchnorm_3")
    fullyConnectedLayer(featureSize,"Name","fc_3")
    leakyReluLayer(leakyReluScale,"Name","leakyrelu_4")
    batchNormalizationLayer("Name","letent space")
    fullyConnectedLayer(1,"Name","fc_4")
    sigmoidLayer("Name","sigmoid")];
lgraph = addLayers(lgraph,tempLayers);

lgraph = connectLayers(lgraph,"Input","Pathways");
lgraph = connectLayers(lgraph,"Input","formattingLayer_1");
lgraph = connectLayers(lgraph,"Pathways","depthcat/in1");
lgraph = connectLayers(lgraph,"formattingLayer_1","depthcat/in2");
lgraph = connectLayers(lgraph,"batchnorm_1","split_first");
lgraph = connectLayers(lgraph,"batchnorm_1","split_reference");
lgraph = connectLayers(lgraph,"split_first","concat/in3");
lgraph = connectLayers(lgraph,"split_reference","gmpool");
lgraph = connectLayers(lgraph,"split_reference","gapool");
lgraph = connectLayers(lgraph,"gmpool","concat/in1");
lgraph = connectLayers(lgraph,"gapool","concat/in2");

netD=dlnetwork(lgraph);
netD=initialize(netD);

end