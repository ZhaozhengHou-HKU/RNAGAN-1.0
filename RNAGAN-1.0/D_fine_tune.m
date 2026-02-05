function [Dnet,Record] = D_fine_tune...
    (Dnet,geneList,positiveRef,negativeRef,numEpochs,validationProportion,useGPU)
%D_DIAGNOSE   Diagnose the given target is the same as the positive
%   reference or not.
%   author: Zhaozheng Hou (George)
%
% [Dnet,Record] = D_fine_tune(Dnet,geneList,positiveRef,negativeRef,echo,
%       validationProportion,useGPU)
% parameter:
%   - Dnet: discriminator network to use (network or the name of trained
%       network, such as "BDPP10" for "the discriminator for bulk-RNA seq
%       data with predefined pathways and using 10 references")
%   - geneList: list of genes (strings or numbers), leave blank and skip
%       the matching if the data already matched the gene list
%   - positiveRef: expression of positive samples, each column is one
%       sample. Better with at least 20 samples.
%   - negativeRef: expression of positive samples, each column is one
%       sample. Better with at least 20 samples.
%   - numEpochs: number of epochs (how many times each sample get used for
%       training). At least 10 epochs.
%   - validationProportion: (optional) proportion of first samples used for
%       validation, 0.3 by default. 0 for training without validation, 1
%       for only validate the network without training.
%   - useGPU: (optional) whether using GPU for the processing or not, true
%       by default
% output:
%   - Dnet: tuned discriminator network
%   - Record: structure of training records: recoreded epochs, networks,
%       validations, and training scores and loss

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
validateattributes(positiveRef,{'numeric'},{'nonnegative','size',[numel(geneList),nan]});
validateattributes(size(positiveRef,2),{'numeric'},{'>=',4});
validateattributes(negativeRef,{'numeric'},{'nonnegative','size',[numel(geneList),nan]});
validateattributes(size(negativeRef,2),{'numeric'},{'>=',4});

validateattributes(numEpochs,{'numeric'},{'integer','>=',10});

validateattributes(validationProportion,{'numeric'},{'>=',0,'<=',1});
if ((validationProportion==1)&&(nargout==1))
    return;
end

if (nargin>=7)
    validateattributes(useGPU,{'numeric','logical'},{'binary'});
    if (isempty(useGPU))
        useGPU=true;
    end
else
    useGPU=true;
end

if (~Dnet.Initialized)
    warning("The network is not yet initialized. Initializing now...");
    Dnet=Dnet.initialize;
end

%% process
% match genes
if (~isnan(geneList))
    [netGeneID,listGeneID] = match_gene_list(geneList);
    temp=zeros(n(1),size(positiveRef,2));
    temp(netGeneID,:)=positiveRef(listGeneID,:);
    positiveRef=dlarray(temp,'SSCB');
    temp=zeros(n(1),size(negativeRef,2));
    temp(netGeneID,:)=negativeRef(listGeneID,:);
    negativeRef=dlarray(temp,'SSCB');
end
if (useGPU)
    try
        positiveRef=gpuArray(positiveRef);
        negativeRef=gpuArray(negativeRef);
    catch
        warning("Error when using GPU, trying with CPU...");
        useGPU=flase;
    end
end

% prepare record
Record.epochs=[(1:9)*floor(numEpochs/10),numEpochs];
Record.networks=cell(10,1);
Record.validations=nan(10,1);
Record.trainingScores=[];
Record.trainingLoss=[];
% train network
if (validationProportion==1)
    Record.epochs=1:numEpochs;
    Record.networks=Dnet;
    Record.trainingScores=nan;
    Record.trainingLoss=nan;

    Record.validations=nan(numEpochs,1);
    for ind=1:numEpochs
        sid=rand(1,n(2),'single');
        Z=positiveRef(:,ceil(end*sid),1,[1,1]);
        Z(:,:,:,2)=negativeRef(:,ceil(end*sid),1,[1,1]);
        Z=repmat(Z,[1,1,1,2]);
        Z(:,1,1,[3,4])=Z(:,1,1,[4,3]);
        Y=predict(Dnet,Z);
        Record.validations(ind)=(Y(1)+Y(2)-Y(3)-Y(4)+2)/4;
    end
    return;
end

addpath("core\");

learnRate = 0.02;
gradientDecayFactor = 0.9;
squaredGradientDecayFactor = 0.999;

trailingAvgD = [];
trailingAvgSqD = [];

iteration = 0;

% training without validation
if (validationProportion==0)
    positiveTr=positiveRef;
    negativeTr=negativeRef;

    nRef=[size(positiveTr,2),size(negativeTr,2)];
    si=max(nRef);
    iterationPerEpoch=ceil(si/10);

    Z=repmat(positiveRef(:,1,1,1),[1,n(2),1,20]);
    for ind=1:numEpochs
        temp=ceil(nRef.*rand(10*iterationPerEpoch,2));
        temp(1:si,1)=randperm(nRef(1),si);
        temp(1:si,2)=randperm(nRef(2),si);

        for ind2=1:iterationPerEpoch
            try
                sid=[temp(10*ind2+(-9:0),1),ceil(nRef(1)*rand(10,n(2)-1))];
                Z(:,:,1,1:2:end)=reshape(positiveTr(:,sid'),[n(1),n(2),1,10]);
                sid=[temp(10*ind2+(-9:0),2),ceil(nRef(2)*rand(10,n(2)-1))];
                Z(:,:,1,2:2:end)=reshape(negativeTr(:,sid'),[n(1),n(2),1,10]);

                [Record.trainingLoss(end+1),Record.trainingScores(end+1),...
                    gradientsD,stateD] = ...
                    dlfeval(@modelLossD,Dnet,Z);
                if (useGPU)
                    Dnet.State = stateD.gather();
                    [Dnet,trailingAvgD,trailingAvgSqD] = adamupdate(Dnet,...
                        gradientsD.gather(),trailingAvgD, trailingAvgSqD, iteration, ...
                        learnRate, gradientDecayFactor, squaredGradientDecayFactor);
                else
                    Dnet.State = stateD;
                    [Dnet,trailingAvgD,trailingAvgSqD] = adamupdate(Dnet,...
                        gradientsD,trailingAvgD, trailingAvgSqD, iteration, ...
                        learnRate, gradientDecayFactor, squaredGradientDecayFactor);
                end
                iteration=iteration+1;
            catch
                warning(num2str(ind,...
                    "An error occured at the %u th epoch. return the latest record."));
                return;
            end
        end

        if (nargout>=2)
            temp=find(ind==Record.epochs);
            if (~isempty(temp))
                Record.networks{temp}=Dnet;
            end
        end
    end
else
    % use the first few for training and the rest for validation, a least 2
    % in each group.
    validationProportion=1-validationProportion;

    if (nargout>=2)
        positiveVa=positiveRef(:,min(end-1,ceil(end*validationProportion)):end);
        negativeVa=negativeRef(:,min(end-1,ceil(end*validationProportion)):end);
        nRef=[size(positiveVa,2),size(negativeVa,2)];
        ZVa=repmat(positiveRef(:,1,1,1),[1,n(2),1,2*sum(nRef)]);
        sid=[(1:nRef(1))',ceil(nRef(1)*rand(nRef(1),n(2)-1))];
        ZVa(:,:,1,1:nRef(1))=reshape(positiveVa(:,sid'),[n(1),n(2),1,nRef(1)]);
        sid=[(1:nRef(2))',ceil(nRef(2)*rand(nRef(2),n(2)-1))];
        ZVa(:,:,1,nRef(1)+(1:nRef(2)))=reshape(negativeVa(:,sid'),[n(1),n(2),1,nRef(2)]);
        ZVa(:,1,1,(sum(nRef)+1):end)=ZVa(:,1,1,1:sum(nRef));
        sid=[ceil(nRef(2)*rand(nRef(1),n(2)-1))];
        ZVa(:,:,1,sum(nRef)+(1:nRef(1)))=reshape(negativeVa(:,sid'),[n(1),n(2),1,nRef(1)]);
        sid=[ceil(nRef(1)*rand(nRef(2),n(2)-1))];
        ZVa(:,:,1,(end-nRef(2)):end)=reshape(positiveVa(:,sid'),[n(1),n(2),1,nRef(2)]);
        if (useGPU)
            ZVa=ZVa.gather();
        end
    end


    positiveTr=positiveRef(:,1:max(2,floor(end*validationProportion)));
    negativeTr=negativeRef(:,1:max(2,floor(end*validationProportion)));

    nRef=[size(positiveTr,2),size(negativeTr,2)];
    si=max(nRef);
    iterationPerEpoch=ceil(si/10);

    Z=repmat(positiveTr(:,1,1,1),[1,n(2),1,20]);
    for ind=1:numEpochs
        temp=ceil(nRef.*rand(10*iterationPerEpoch,2));
        temp(1:si,1)=randperm(nRef(1),si);
        temp(1:si,2)=randperm(nRef(2),si);

        for ind2=1:iterationPerEpoch
            try
                sid=[temp(10*ind2+(-9:0),1),ceil(nRef(1)*rand(10,n(2)-1))];
                Z(:,:,1,1:2:end)=reshape(positiveTr(:,sid'),[n(1),n(2),1,10]);
                sid=[temp(10*ind2+(-9:0),2),ceil(nRef(2)*rand(10,n(2)-1))];
                Z(:,:,1,2:2:end)=reshape(negativeTr(:,sid'),[n(1),n(2),1,10]);

                [Record.trainingLoss(end+1),Record.trainingScores(end+1),...
                    gradientsD,stateD] = ...
                    dlfeval(@modelLossD,Dnet,Z);
                if (useGPU)
                    Dnet.State = stateD.gather();
                    [Dnet,trailingAvgD,trailingAvgSqD] = adamupdate(Dnet,...
                        gradientsD.gather(),trailingAvgD, trailingAvgSqD, iteration, ...
                        learnRate, gradientDecayFactor, squaredGradientDecayFactor);
                else
                    Dnet.State = stateD;
                    [Dnet,trailingAvgD,trailingAvgSqD] = adamupdate(Dnet,...
                        gradientsD,trailingAvgD, trailingAvgSqD, iteration, ...
                        learnRate, gradientDecayFactor, squaredGradientDecayFactor);
                end
                iteration=iteration+1;
            catch
                warning(num2str(ind,...
                    "An error occured at the %u th epoch. return the latest record."));
                return;
            end
        end

        if (nargout>=2)
            temp=find(ind==Record.epochs);
            if (~isempty(temp))
                Record.networks{temp}=Dnet;
                Y=predict(Dnet,ZVa);
                Y=Y(:);
                Y((end/2+1):end)=1-Y((end/2+1):end);
                Record.validations(temp)=mean(Y);
            end
        end
    end
end

end