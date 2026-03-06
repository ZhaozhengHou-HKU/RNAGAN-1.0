rng(1);

numGene=size(pathwayMatrix,2);
numCases=numel(RefCaseWise);

numEpochs = 10;
learnRate = 0.04;%later 0.01;
gradientDecayFactor = 0.9;
squaredGradientDecayFactor = 0.999;
numBatch=50;
ValidationFrequency = 2;
DsRecord=cell(3,3,numEpochs/ValidationFrequency);

for RefNum10=1:3

    numRef=10*RefNum10;
    for TypeID=1:3
        netD=Ds{TypeID,RefNum10};

        trailingAvgG = [];
        trailingAvgSqG = [];
        trailingAvgD = [];
        trailingAvgSqD = [];

        iteration = 0;

        % Loop over epochs.
        for epoch = 1:numEpochs
            % Reset and shuffle datastore.
            ref=rand(numCases,1)*numBatch;

            ZS=ZZ_SC(:,:,epoch,:);
            % Loop over mini-batches.
            for mbq=1:numBatch
                iteration = iteration + 1;

                disp([numRef, TypeID, iteration]);

                Z = ZS(:,1:(numRef+1),:,(ref<mbq)&(ref>mbq-1));
                % if (mod(iteration,10)==1)
                %     Z(:,:,:,end+(1:39))=ZZ_B(:,1:(numRef+1),iteration,:);
                % end
                [~,~,gradientsD,stateD] = ...
                    dlfeval(@modelLossD,netD,gpuArray(Z));
                %gradientsD=extractdata(gradientsD);
                %stateD=extractdata(stateD);
                netD.State = stateD;
                [netD,trailingAvgD,trailingAvgSqD] = adamupdate(netD, gradientsD, ...
                    trailingAvgD, trailingAvgSqD, iteration, ...
                    learnRate, gradientDecayFactor, squaredGradientDecayFactor);
            end

            if (mod(epoch,ValidationFrequency)==0)
                DsRecord{TypeID,RefNum10,epoch/ValidationFrequency}=netD;
            end
        end
        Ds{TypeID,RefNum10}=netD;
    end
end

save('Step1_SC_D.mat','Ds','DsRecord');