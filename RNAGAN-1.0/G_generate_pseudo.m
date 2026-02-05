function [pseudo,pseudoNet] = G_generate_pseudo...
    (Gnet,geneList,reference,numberOfPseudo,useGPU)
%G_GENERATE_PSEUDO
%   Generate pseudo data according to the given references.
%   author: Zhaozheng Hou (George)
%
% [pseudo,pseudoNet] = G_generate_pseudo(Gnet,geneList,reference,
%       numberOfPseudo,useGPU)
% parameter:
%   - Gnet: generator network to use (network or the name of trained
%       network, such as "BGPP10" for "the generator for bulk-RNA seq
%       data with predefined pathways and using 10 references")
%   - geneList: list of genes (strings or numbers), leave blank and skip
%       the matching if the data already matched the gene list
%   - reference: expression of smples, each column is one sample. At least
%       2 samples.
%   - numberOfPseudo: number of pseudo samples to generate, 1 by default.
%   - useGPU: (optional) whether using GPU for the processing or not, false
%       by default.
% output:
%   - pseudo: data with the same gene list as reference, each column is one
%       pseudo sample. (unmatched genes get NA)
%   - pseudoNet: data with the gene list same as the network.

%% validate
if (isa(Gnet,"dlnetwork"))
    validateattributes(Gnet,{'dlnetwork'},{'scalar'});
else
    validateattributes(Gnet,{'string','char'},{'scalartext'});
    Gnet=load("core\trainedG.mat",string(Gnet));
    Gnet=struct2cell(Gnet);
    Gnet=Gnet{1};
end
n=Gnet.getLayer(Gnet.InputNames{1}).InputSize;

if (isempty(geneList))
    geneList=nan(n(1),1);
else
    validateattributes(geneList,{'numeric','char','string'},{'vector'});
end
validateattributes(reference,{'numeric'},{'nonnegative','size',[numel(geneList),nan]});
validateattributes(size(reference,2),{'numeric'},{'>=',2});

if (nargin>=4)
    validateattributes(numberOfPseudo,{'numeric'},{'integer','positive'});
    if (isempty(numberOfPseudo))
        numberOfPseudo=1;
    end
else
    numberOfPseudo=1;
end

if (nargin>=5)
    validateattributes(useGPU,{'numeric','logical'},{'binary'});
    if (isempty(useGPU))
        useGPU=false;
    end
else
    useGPU=false;
end

if (~Gnet.Initialized)
    warning("The network is not yet initialized. Initializing now...");
    Gnet=Gnet.initialize;
end

%% process
sizeOfReference=size(reference,2);

% match genes
if (~isnan(geneList))
    [netGeneID,listGeneID] = match_gene_list(geneList);
    temp=zeros(n(1),sizeOfReference);
    temp(netGeneID,:)=reference(listGeneID,:);
    reference=temp;
end
valid=any(reference>0,2);

% prepare references
if ((numberOfPseudo==1)&&(sizeOfReference<=n(2)))
    % only one vector for the given reference
    % no need to repeat
    sid=repmat(1:sizeOfReference,n(2));
    reference=dlarray(reference(:,sid(1:n(2))),'SSCB');
else
    sid=ceil(sizeOfReference*rand(numberOfPseudo,n(2)));
    reference=reshape(reference(:,sid'),[n(1),n(2),1,numberOfPseudo]);
    reference=dlarray(reference,'SSCB');
end

% calculate pseudo data
if (useGPU)
    try
        pseudoNet=predict(Gnet,gpuArray(reference)).gather.extractdata;
    catch
        warning("Error when using GPU, trying with CPU...");
         pseudoNet=predict(Gnet,reference).extractdata;
    end
end

% normalization
temp=log(pseudoNet+1);
for ind=1:numberOfPseudo
    temp(:,1,1,ind)=exp(temp(:,1,1,ind)-...
        median(mean(temp(valid,1,1,ind))-mean(temp(valid,2:end,1,ind,2))))-1;
end
pseudoNet=permute(temp(:,1,1,:),[1,4,2,3]);

% reformating
if (isnan(geneList))
    pseudo=pseudoNet;
else
    pseudo=nan(numel(geneList),numberOfPseudo);
    pseudo(listGeneID,:)=pseudoNet(netGeneID,:);
end
end