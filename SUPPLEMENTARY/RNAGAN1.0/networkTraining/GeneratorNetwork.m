function netG = GeneratorNetwork(pathwayMask,numRef)
%provide pathway in [pathway,genes]
%learnable pathways with all true
%no pathway with an empty matrix with 0 rows, or as the number of genes

if (isscalar(pathwayMask))
    numPathways=0;
    numGene=pathwayMask;
else
    [numPathways,numGene]=size(pathwayMask);
end

leakyReluScale=0.01; %default value
featureSize=16;

netG = dlnetwork;

tempNet = imageInputLayer([numGene numRef 1],"Name","Input","Normalization","none");
netG = addLayers(netG,tempNet);

if (numPathways>0)
    tempNet = convolution2dLayer([numGene 1],numPathways,'Name','Pathways');
    if (~all(pathwayMask,'all')) % load pathway info
        tempNet.WeightLearnRateFactor=0;
        tempNet.Weights=single(permute(pathwayMask,[2,3,4,1]));
    end
    netG = addLayers(netG,tempNet);
end

tempNet = reshapeLayer("formattingLayer_1",[3 2 1 4],"SSCB");
netG = addLayers(netG,tempNet);

if (numPathways>0)
    tempNet = [
        depthConcatenationLayer(2,"Name","depthcat")
        leakyReluLayer(leakyReluScale,"Name","leakyrelu_1")
        batchNormalizationLayer("Name","batchnorm_1")];
else
    tempNet = [
        leakyReluLayer(0.01,"Name","leakyrelu_1")
        batchNormalizationLayer("Name","batchnorm_1")];
end
netG = addLayers(netG,tempNet);

tempNet = globalMaxPooling2dLayer("Name","gmpool");
netG = addLayers(netG,tempNet);

tempNet = globalAveragePooling2dLayer("Name","gapool");
netG = addLayers(netG,tempNet);

tempNet = globalAveragePooling2dLayer("Name","gapool_2");
netG = addLayers(netG,tempNet);

tempNet = [
    concatenationLayer(2,2,"Name","concat_1")
    reshapeLayer("formattingLayer_2",[3 2 1 4],"SSCB")
    fullyConnectedLayer(16*featureSize,"Name","fc_1")
    leakyReluLayer(leakyReluScale,"Name","leakyrelu_2")
    batchNormalizationLayer("Name","batchnorm_2")
    dropoutLayer(0.5,"Name","dropout")];
netG = addLayers(netG,tempNet);

tempNet = [
    fullyConnectedLayer(4*featureSize,"Name","fc_2")
    leakyReluLayer(leakyReluScale,"Name","leakyrelu_3")
    batchNormalizationLayer("Name","batchnorm_3")];
netG = addLayers(netG,tempNet);

tempNet = [
    fullyConnectedLayer(1*featureSize,"Name","fc_3")
    leakyReluLayer(leakyReluScale,"Name","leakyrelu_4")
    batchNormalizationLayer("Name","letent space")
    fullyConnectedLayer(4*featureSize,"Name","fc_4")
    leakyReluLayer(leakyReluScale,"Name","leakyrelu_5")
    batchNormalizationLayer("Name","batchnorm_4")];
netG = addLayers(netG,tempNet);

tempNet = [
    concatenationLayer(1,2,"Name","concat_2")
    fullyConnectedLayer(16*featureSize,"Name","fc_5")
    leakyReluLayer(leakyReluScale,"Name","leakyrelu_6")
    batchNormalizationLayer("Name","batchnorm_5")];
netG = addLayers(netG,tempNet);

tempNet = [
    concatenationLayer(1,2,"Name","concat_3")
    fullyConnectedLayer(numGene,"Name","fc_GeneLevel")
    reluLayer("Name","GeneLevelOutput")
    reshapeLayer("formattingLayerasdf",[3 4 1 2],"SSCB")];
netG = addLayers(netG,tempNet);

tempNet = [
    multiplicationLayer(2,"Name","multiplication")
    reshapeLayer("formattingLayer_1_1",[3 2 1 4],"SSCB")];
netG = addLayers(netG,tempNet);

tempNet = concatenationLayer(2,2,"Name","concat");
netG = addLayers(netG,tempNet);

if (numPathways>0)
    netG = connectLayers(netG,"Input","Pathways");
    netG = connectLayers(netG,"Pathways","depthcat/in1");
    netG = connectLayers(netG,"formattingLayer_1","depthcat/in2");
    netG = connectLayers(netG,"Input","formattingLayer_1");
else
    netG = connectLayers(netG,"Input","formattingLayer_1");
    netG = connectLayers(netG,"formattingLayer_1","leakyrelu_1");
end
netG = connectLayers(netG,"Input","concat/in2");
netG = connectLayers(netG,"formattingLayer_1","gapool_2");
netG = connectLayers(netG,"batchnorm_1","gmpool");
netG = connectLayers(netG,"batchnorm_1","gapool");
netG = connectLayers(netG,"gmpool","concat_1/in1");
netG = connectLayers(netG,"gapool","concat_1/in2");
netG = connectLayers(netG,"gapool_2","multiplication/in1");
netG = connectLayers(netG,"dropout","fc_2");
netG = connectLayers(netG,"dropout","concat_3/in2");
netG = connectLayers(netG,"batchnorm_3","fc_3");
netG = connectLayers(netG,"batchnorm_3","concat_2/in2");
netG = connectLayers(netG,"batchnorm_4","concat_2/in1");
netG = connectLayers(netG,"batchnorm_5","concat_3/in1");
netG = connectLayers(netG,"formattingLayerasdf","multiplication/in2");
netG = connectLayers(netG,"formattingLayer_1_1","concat/in1");
netG = initialize(netG);
end