function Features = D_extract_eatures...
    (Dnet,geneList,positiveRef,negativeRef,featureOrder)
%D_EXTRACT_EATURES   Extract features of given feature order.
%   author: Zhaozheng Hou (George)
%
% Features = D_extract_eatures(Dnet,geneList,positiveRef,negativeRef,
%       featureOrder)
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
%   - featureOrder: order of features to extract. 0/1/2
% output:
%   - Features: structure of extracted features

warning off;

%% validate
if (isa(Dnet,"dlnetwork"))
    validateattributes(Dnet,{'dlnetwork'},{'scalar'});
else
    validateattributes(Dnet,{'string','char'},{'scalartext'});
    Dnet=load("core\"+string(Dnet)+".mat",string(Dnet));
    Dnet=struct2cell(Dnet);
    Dnet=Dnet{1};
end
n=Dnet.getLayer(Dnet.InputNames{1}).InputSize;

if (isempty(geneList))
    geneList=nan(n(1),1);
    netGeneID=1:n(1);
    listGeneID=netGeneID;
else
    validateattributes(geneList,{'numeric','char','string'},{'vector'});
end
validateattributes(positiveRef,{'numeric'},{'nonnegative','size',[numel(geneList),nan]});
validateattributes(size(positiveRef,2),{'numeric'},{'>=',2});
validateattributes(negativeRef,{'numeric'},{'nonnegative','size',[numel(geneList),nan]});
validateattributes(size(negativeRef,2),{'numeric'},{'>=',2});

validateattributes(featureOrder,{'numeric'},{'integer','>=',0,'<=',2});

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
    positiveRef=temp;
    temp=zeros(n(1),size(negativeRef,2));
    temp(netGeneID,:)=negativeRef(listGeneID,:);
    negativeRef=temp;
end

nRef=size(positiveRef,2);
%[positive,positive]
sid=[(1:nRef)',ceil(nRef*rand(nRef,n(2)-1))];
Z1=reshape(positiveRef(:,sid'),[n(1),n(2),1,nRef]);
%Z1=dlarray(Z1,'SSCB');
sid=ceil(size(negativeRef,2)*rand(nRef,n(2)-1));
%[positive,negative]
Z2=Z1;
Z2(:,2:end,1,:)=reshape(negativeRef(:,sid'),[n(1),n(2)-1,1,nRef]);

Dnet=initialize(Dnet.removeLayers('sigmoid'));

Features.AUC=predict(Dnet,Z1);
temp=predict(Dnet,Z2);
Features.AUC=mean(Features.AUC(:)>temp(:)',"all");

usingGPU=isgpuarray(Features.AUC);
if (usingGPU)
    Features.AUC=Features.AUC.gather();
end
fl=(Features.AUC<0.5);
if (fl)
    Features.AUC=1-Features.AUC;
end

switch (featureOrder)
    case 0 % occlusionSensitivity
        temp=zeros(n(1),n(2),1,1);
        temp(:,1)=1;
        temp=predict(Dnet,temp,Outputs='formattingLayer_2');
        numPathways=size(temp,1)-n(1);
        n(1)=size(temp,1);
        if (numPathways>0)
            Z1=predict(Dnet,Z1,Outputs='depthcat');
            Z2=predict(Dnet,Z2,Outputs='depthcat');
            Dnet=Dnet.removeLayers('Input');
            Dnet=Dnet.removeLayers('Pathways');
            Dnet=Dnet.removeLayers('depthcat');
            Dnet=Dnet.connectLayers('formattingLayer_1','leakyrelu_1');
            Dnet=Dnet.addInputLayer(imageInputLayer([n(1),n(2)],Normalization="none"));
            Dnet=Dnet.initialize;
            Z1=permute(Z1,[3,2,1,4]);
            Z2=permute(Z2,[3,2,1,4]);
        end

        Features.occlusionSensitivity=zeros(nRef,n(1));
        Zt1=Z1;
        Zt2=Z2;
        valid=any(Z1(:,:,1,:)>quantile(positiveRef,0.25,'all'),[2,4]);
        for ind=1:n(1)
            if (valid(ind))
                disp(num2str([ind,n(1)],"Processing %u out of %u features..."));

                Zt2(ind,1,1,:)=0;
                Features.occlusionSensitivity(:,ind)=predict(Dnet,Zt2);
                Zt2(ind,1,1,:)=Z2(ind,1,1,:);

                Zt1(ind,1,1,:)=0;
                Features.occlusionSensitivity(:,ind)=Features.occlusionSensitivity(:,ind)...
                    -predict(Dnet,Zt1);
                Zt1(ind,1,1,:)=Z1(ind,1,1,:);
            end
        end
        Features.occlusionSensitivity=mean(Features.occlusionSensitivity,1)'...
            +mean(predict(Dnet,Z1))-mean(predict(Dnet,Z2));
        Features.occlusionSensitivity(~valid)=0;
        if (usingGPU)
            Features.occlusionSensitivity=Features.occlusionSensitivity.gather();
        end
        if (fl)
            Features.occlusionSensitivity=-Features.occlusionSensitivity;
        end

        if (numPathways>0)
            Features.occlusionSensitivity_Genes=...
                Features.occlusionSensitivity((numPathways+1):end);
            temp=nan(numel(geneList),1);
            temp(listGeneID)=Features.occlusionSensitivity_Genes(netGeneID);
            Features.occlusionSensitivity_Genes=temp;

            Features.occlusionSensitivity_Pathways=...
                Features.occlusionSensitivity(1:numPathways);
        end



    case 1 % gradCAM, log2FC, MannWhitneyU, and gradient
        temp=zeros(n(1),n(2),1,1);
        temp(:,1)=1;
        temp=predict(Dnet,temp,Outputs='formattingLayer_2');
        numPathways=size(temp,1)-n(1);
        temp=sum(temp,1);
        [~,targetID]=max(abs(temp-mean(temp)));
        gradCAMscores=zeros(size(temp,1),1);

        if (numPathways>0)
            Z1=predict(Dnet,Z1,Outputs='depthcat');
            Z2=predict(Dnet,Z2,Outputs='depthcat');
            Dnet=Dnet.removeLayers('Input');
            Dnet=Dnet.removeLayers('Pathways');
            Dnet=Dnet.removeLayers('depthcat');
            Dnet=Dnet.connectLayers('formattingLayer_1','leakyrelu_1');
            Dnet=Dnet.addInputLayer(imageInputLayer([n(1)+numPathways,n(2)],Normalization="none"));
            Dnet=Dnet.initialize;
            Z1=dlarray(permute(Z1,[3,2,1,4]),'SSCB');
            Z2=dlarray(permute(Z2,[3,2,1,4]),'SSCB');
        end

        for ind=1:nRef
            temp=gradCAM(Dnet,Z1(:,:,1,ind),@(x)x,FeatureLayer="formattingLayer_2",OutputUpsampling="none");
            gradCAMscores=gradCAMscores+temp(:,targetID);
            temp=gradCAM(Dnet,Z1(:,:,1,ind),@(x)-x,FeatureLayer="formattingLayer_2",OutputUpsampling="none");
            gradCAMscores=gradCAMscores-temp(:,targetID);
            temp=gradCAM(Dnet,Z2(:,:,1,ind),@(x)-x,FeatureLayer="formattingLayer_2",OutputUpsampling="none");
            gradCAMscores=gradCAMscores+temp(:,targetID);
            temp=gradCAM(Dnet,Z2(:,:,1,ind),@(x)x,FeatureLayer="formattingLayer_2",OutputUpsampling="none");
            gradCAMscores=gradCAMscores-temp(:,targetID);
        end
        if (fl)
            gradCAMscores=-gradCAMscores;
        end

        if (numPathways>0)
            Features.gradCAM_Genes=gradCAMscores/(2*nRef);
            Features.gradCAM_Pathways=Features.gradCAM_Genes(1:numPathways);
            Features.gradCAM_Genes=Features.gradCAM_Genes((numPathways+1):end);
            temp=nan(numel(geneList),1);
            temp(listGeneID)=Features.gradCAM_Genes(netGeneID);
            Features.gradCAM_Genes=temp;

            Features.log2FC_Genes=log2(mean(positiveRef,2)./mean(negativeRef,2));
            if (usingGPU)
                Features.log2FC_Genes=Features.log2FC_Genes.gather();
            end

            Features.MannWhitneyU_Genes=nan(numel(geneList),1);
            for ind=1:numel(geneList)
                Features.MannWhitneyU_Genes(ind)=ranksum(positiveRef(ind,:),negativeRef(ind,:));
            end

            gradient=nan(n(1),1);
            Zt1=gpuArray(Z1);
            Zt2=gpuArray(Z2);
            for ind=18584:n(1)
                disp(num2str([ind,n(1)],"Processing %u out of %u genes..."));

                Zt1(ind,1,1,:)=Z1(ind,1,1,:)*1.05;
                temp=predict(Dnet,Zt1);
                Zt1(ind,1,1,:)=Z1(ind,1,1,:)*0.95;
                temp=temp-predict(Dnet,Zt1);
                Zt1(ind,1,1,:)=Z1(ind,1,1,:);

                Zt2(ind,1,1,:)=Z2(ind,1,1,:)*1.05;
                temp=temp-predict(Dnet,Zt2);
                Zt2(ind,1,1,:)=Z2(ind,1,1,:)*0.95;
                temp=temp+predict(Dnet,Zt2);
                Zt2(ind,1,1,:)=Z2(ind,1,1,:);

                gradient(ind)=mean(temp)*5; % *10/2
            end
            if (fl)
                gradient=-gradient;
            end

            Features.gradient_Genes=nan(numel(geneList),1);
            Features.gradient_Genes(listGeneID)=gradient(netGeneID);

            positiveRef=permute(predict(Dnet,positiveRef,Outputs='depthcat'),[3,2,1,4]);
            negativeRef=permute(predict(Dnet,negativeRef,Outputs='depthcat'),[3,2,1,4]);
            Features.log2FC_Pathways=log2(mean(positiveRef(1:numPathways,:),2)./mean(negativeRef(1:numPathways,:),2));

            Z1=predict(Dnet,Z1,Outputs='depthcat');
            Z2=predict(Dnet,Z2,Outputs='depthcat');
            Dnet=Dnet.removeLayers('Input');
            Dnet=Dnet.removeLayers('Pathways');
            Dnet=Dnet.removeLayers('depthcat');
            Dnet=Dnet.connectLayers('formattingLayer_1','leakyrelu_1');
            Dnet=Dnet.addInputLayer(imageInputLayer([n(1)+numPathways,n(2)],Normalization="none"));
            Dnet=Dnet.initialize;
            Z1=permute(Z1,[3,2,1,4]);
            Z2=permute(Z2,[3,2,1,4]);

            Features.MannWhitneyU_Pathways=nan(numPathways,1);
            Features.gradient_Pathways=nan(numPathways,1);

            Zt1=Z1;
            Zt2=Z2;
            for ind=1:numPathways
                disp(num2str([ind,numPathways],"Processing %u out of %u pathways..."));


                %Features.MannWhitneyU_Pathways(ind)=ranksum(positiveRef(ind,:),negativeRef(ind,:));

                Zt1(ind,1,1,:)=Z1(ind,1,1,:)*1.05;
                temp=predict(Dnet,Zt1);
                Zt1(ind,1,1,:)=Z1(ind,1,1,:)*0.95;
                temp=temp-predict(Dnet,Zt1);
                Zt1(ind,1,1,:)=Z1(ind,1,1,:);

                Zt2(ind,1,1,:)=Z2(ind,1,1,:)*1.05;
                temp=temp-predict(Dnet,Zt2);
                Zt2(ind,1,1,:)=Z2(ind,1,1,:)*0.95;
                temp=temp+predict(Dnet,Zt2);
                Zt2(ind,1,1,:)=Z2(ind,1,1,:);

                Features.gradient_Pathways(ind)=mean(temp)*5;
            end
            if (fl)
                Features.gradient_Pathways=-Features.gradient_Pathways;
            end
        else
            Features.gradCAM_Genes=gradCAMscores/(2*nRef);
            temp=nan(numel(geneList),1);
            temp(listGeneID)=Features.gradCAM_Genes(netGeneID);
            Features.gradCAM_Genes=temp;

            Features.log2FC_Genes=log2(mean(positiveRef,2)./mean(negativeRef,2));

            Features.MannWhitneyU_Genes=nan(numel(geneList),1);
            for ind=1:numel(geneList)
                Features.MannWhitneyU_Genes(ind)=ranksum(positiveRef(ind,:),negativeRef(ind,:));
            end

            gradient=nan(n(1),1);
            Zt1=Z1;
            Zt2=Z2;
            for ind=1:n(1)
                disp(num2str([ind,n(1)],"Processing %u out of %u features..."));

                Zt1(ind,1,1,:)=Z1(ind,1,1,:)*1.05;
                temp=predict(Dnet,Zt1);
                Zt1(ind,1,1,:)=Z1(ind,1,1,:)*0.95;
                temp=temp-predict(Dnet,Zt1);
                Zt1(ind,1,1,:)=Z1(ind,1,1,:);

                Zt2(ind,1,1,:)=Z2(ind,1,1,:)*1.05;
                temp=temp-predict(Dnet,Zt2);
                Zt2(ind,1,1,:)=Z2(ind,1,1,:)*0.95;
                temp=temp+predict(Dnet,Zt2);
                Zt2(ind,1,1,:)=Z2(ind,1,1,:);

                gradient(ind)=mean(temp)*5;
            end
            if (fl)
                gradient=-gradient;
            end

            Features.gradient_Genes=nan(numel(geneList),1);
            Features.gradient_Genes(listGeneID)=gradient(netGeneID);
        end

    case 2 % 1&2 order gradient and R2
        temp=zeros(n(1),n(2),1,1);
        temp(:,1)=1;
        temp=predict(Dnet,temp,Outputs='formattingLayer_2');
        numPathways=size(temp,1)-n(1);
        n(1)=size(temp,1);
        if (numPathways>0)
            Z1=predict(Dnet,Z1,Outputs='depthcat');
            Z2=predict(Dnet,Z2,Outputs='depthcat');
            Dnet=Dnet.removeLayers('Input');
            Dnet=Dnet.removeLayers('Pathways');
            Dnet=Dnet.removeLayers('depthcat');
            Dnet=Dnet.connectLayers('formattingLayer_1','leakyrelu_1');
            Dnet=Dnet.addInputLayer(imageInputLayer([n(1),n(2)],Normalization="none"));
            Dnet=Dnet.initialize;
            Z1=permute(Z1,[3,2,1,4]);
            Z2=permute(Z2,[3,2,1,4]);
        end

        % number of turbulence reference for score differenation
        numDiffTrials=100;
        Z12=repmat(Z1,[1,1,1,2]);
        Z12(:,:,:,(end/2+1):end)=Z2;

        diffScore=nan(size(Z12,4),numDiffTrials);
        delta=(rand(size(Z12,1),numDiffTrials,'single')-0.5)/10;
        delta=delta-mean(delta,2);
        Zt=Z12;
        for ind=1:numDiffTrials
            Zt(:,1,:,:)=Z12(:,1,:,:).*(1+delta(:,ind));
            temp=predict(Dnet,Zt);
            diffScore(:,ind)=temp(:);
        end
        temp=predict(Dnet,Z12);
        diffScore=diffScore-temp(:);
        diffScore((end/2+1):end,:)=-diffScore((end/2+1):end,:);
        diffScore=mean(diffScore,1)';

        valid=any(Z12(:,:,1,:)>quantile(positiveRef,0.25,'all'),[2,4]);
        delta=delta';
        xx=single([ones(numDiffTrials,1),zeros(numDiffTrials,3)]);
        % evaluate
        gradients=zeros(n(1));
        R2=gradients;
        warning('off');
        try
            diffScore=diffScore.gather();
        catch
        end
        for ind=1:n(1)
            if (valid(ind))
                disp(num2str([ind,n(1)],"Processing %u out of %u features..."));
                xx(:,2)=delta(:,ind);
                for ind2=(ind+1):n(1)
                    if (valid(ind2))
                        xx(:,3)=delta(:,ind2);
                        xx(:,4)=xx(:,2).*xx(:,3);
                        % temp=xx\diffScore;
                        temp=diffScore'*xx/(xx'*xx);
                        gradients(ind,ind)=gradients(ind,ind)+temp(2);
                        gradients(ind2,ind2)=gradients(ind2,ind2)+temp(3);
                        gradients(ind,ind2)=temp(4);
                        R2(ind,ind2)=sum((diffScore-xx(:,1:3)*temp(1:3)').^2);
                        R2(ind,ind2)=max(0,1-sum((diffScore-xx*temp').^2)/R2(ind,ind2));
                    end
                end
            end
        end
        SST=sum(diffScore.^2);
        factor=sum(valid)-1;
        for ind=1:n(1)
            if (valid(ind))
                gradients(ind,ind)=gradients(ind,ind)/factor;
                temp=diffScore-gradients(ind,ind).*delta(:,ind);
                R2(ind,ind)=(1-sum((temp-mean(temp)).^2)/SST)/2;
                gradients(ind,ind)=gradients(ind,ind)/2; % devide by half for next step
            end
        end
        Features.gradients=gradients+gradients';
        Features.R2=R2+R2';
        if (fl)
            Features.gradients=-Features.gradients;
        end

        if (numPathways>0)
            Features.gradients_Pathways=Features.gradients(1:numPathways,1:numPathways);
            Features.R2_Pathways=Features.R2(1:numPathways,1:numPathways);

            Features.gradients_GenePathways=nan(numel(geneList),numPathways);
            Features.R2_GenePathways=Features.gradients_GenePathways;
            Features.gradients_GenePathways(listGeneID,:)=...
                Features.gradients(numPathways+netGeneID,1:numPathways);
            Features.R2_GenePathways(listGeneID,:)=...
                Features.R2(numPathways+netGeneID,1:numPathways);
        end
        Features.gradients_Genes=nan(numel(geneList));
        Features.R2_Genes=Features.gradients_Genes;
        Features.gradients_Genes(listGeneID,listGeneID)=...
            Features.gradients(numPathways+netGeneID,numPathways+netGeneID);
        Features.R2_Genes(listGeneID,listGeneID)=...
            Features.R2(numPathways+netGeneID,numPathways+netGeneID);
end


end