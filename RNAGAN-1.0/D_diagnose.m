function P = D_diagnose...
    (Dnet,geneList,target,positiveRef,useGPU,negativeRef)
%D_DIAGNOSE   Diagnose the given target is the same as the positive
%   reference or not.
%   author: Zhaozheng Hou (George)
%
% P = D_diagnose(Dnet,geneList,target,positiveRef,useGPU,negativeRef)
% parameter:
%   - Dnet: discriminator network to use (network or the name of trained
%       network, such as "BDPP10" for "the discriminator for bulk-RNA seq
%       data with predefined pathways and using 10 references")
%   - geneList: list of genes (strings or numbers), leave blank and skip
%       the matching if the data already matched the gene list
%   - target: expression of targets, each column is one sample
%   - positiveRef: expression of positive samples, each column is one
%       sample. Better with at least 20 samples.
%   - useGPU: (optional) whether using GPU for the processing or not, false
%       by default
%   - negativeRef: (optional) expression of positive samples, each column
%       is one sample
% output:
%   - P: Probability of the target is of the same type as the positive

%% validate
if (isa(Dnet,"dlnetwork"))
    validateattributes(Dnet,{'dlnetwork'},{'scalar'});
else
    validateattributes(Dnet,{'string','char'},{'scalartext'});
    Dnet=load("core\trainedD.mat",string(Dnet));
    Dnet=struct2cell(Dnet);
    Dnet=Dnet{1};
end
n=Dnet.getLayer(Dnet.InputNames{1}).InputSize;

if (isempty(geneList))
    geneList=nan(n(1),1);
else
    validateattributes(geneList,{'numeric','char','string'},{'vector'});
end
validateattributes(target,{'numeric'},{'nonnegative','size',[numel(geneList),nan]});
validateattributes(positiveRef,{'numeric'},{'nonnegative','size',[numel(geneList),nan]});
validateattributes(size(positiveRef,2),{'numeric'},{'>=',2});

if (nargin>=5)
    validateattributes(useGPU,{'numeric','logical'},{'binary'});
    if (isempty(useGPU))
        useGPU=false;
    end
else
    useGPU=false;
end

if (nargin>=6)
    validateattributes(negativeRef,{'numeric'},{'nonnegative','size',[numel(geneList),nan]});
    validateattributes(size(negativeRef,2),{'numeric'},{'>=',2});
else
    negativeRef=[];
end

if (~Dnet.Initialized)
    warning("The network is not yet initialized. Initializing now...");
    Dnet=Dnet.initialize;
end

%% process
% match genes
if (~isnan(geneList))
    [netGeneID,listGeneID] = match_gene_list(geneList);
    temp=zeros(n(1),size(target,2));
    temp(netGeneID,:)=target(listGeneID,:);
    target=temp;
    temp=zeros(n(1),size(positiveRef,2));
    temp(netGeneID,:)=positiveRef(listGeneID,:);
    positiveRef=temp;
    if (~isempty(negativeRef))
        temp=zeros(n(1),size(negativeRef,2));
        temp(netGeneID,:)=negativeRef(listGeneID,:);
        negativeRef=temp;
    end
end

% prepare references
ntrials=100; %number of trials for evaluation
sid=rand([1,n(2),1,ntrials],"single");

positiveTest=reshape(positiveRef(:,ceil(end*sid(:))),[n(1),n(2),1,ntrials]);
positiveTest=dlarray(positiveTest,'SSCB');
if (useGPU)
    positiveTest=gpuArray(positiveTest);
end
if (~isempty(negativeRef))
    negativeTest=reshape(negativeRef(:,ceil(end*sid(:))),[n(1),n(2),1,ntrials]);
    negativeTest=dlarray(negativeTest,'SSCB');
    if (useGPU)
        negativeTest=gpuArray(negativeTest);
    end
end

% calculate P
P=nan(size(target,2),1);
try
    if (isempty(negativeRef))
        for ind=1:size(target,2)
            disp(num2str([ind,size(target,2)],"Processing %u out of %u targets..."));

            positiveTest(:,1,1,:)=repmat(target(:,ind),[1,1,1,ntrials]);
            temp=predict(Dnet,positiveTest);
            P(ind)=mean(temp>0.5);
        end
    else
        for ind=1:size(target,2)
            disp(num2str([ind,size(target,2)],"Processing %u out of %u targets..."));

            positiveTest(:,1,1,:)=repmat(target(:,ind),[1,1,1,ntrials]);
            temp=predict(Dnet,positiveTest);
            negativeTest(:,1,1,:)=repmat(target(:,ind),[1,1,1,ntrials]);
            temp2=predict(Dnet,negativeTest);
            P(ind)=mean(temp(:)>temp2(:)',"all");
        end
    end
catch ME % try to solve if GPU is the reason of exception
    if (useGPU)
        if (isempty(negativeRef))
            positiveTest=positiveTest.gather;
            for ind=1:size(target,2)
                positiveTest(:,1,1,:)=repmat(target(:,ind),[1,1,1,ntrials]);
                temp=predict(Dnet,positiveTest);
                P(ind)=mean(temp>0.5);
            end
        else
            positiveTest=positiveTest.gather;
            negativeTest=negativeTest.gather;
            for ind=1:size(target,2)
                positiveTest(:,1,1,:)=repmat(target(:,ind),[1,1,1,ntrials]);
                temp=predict(Dnet,positiveTest);
                negativeTest(:,1,1,:)=repmat(target(:,ind),[1,1,1,ntrials]);
                temp2=predict(Dnet,negativeTest);
                P(ind)=mean(temp(:)>temp2(:)',"all");
            end
        end
        warning("Error when using GPU, and problem solved by using CPU.")
    else
        rethrow(ME)
    end
end
end