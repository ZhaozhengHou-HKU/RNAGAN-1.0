rng(2);

numGene=size(pathwayMatrix,2);
numCases=numel(RefCaseWise);

numEpochs = 10;
learnRate = 0.02;%generator 0.04;
gradientDecayFactor = 0.9;
squaredGradientDecayFactor = 0.999;
numBatch=50;
ValidationFrequency = 2;

DsRecord=cell(3,3,numEpochs/ValidationFrequency);
GsRecord=cell(3,3,numEpochs/ValidationFrequency);

for RefNum10=1:3

    numRef=10*RefNum10;
    for TypeID=1:3
        netD=Ds{TypeID,RefNum10};
        netG=Gs{TypeID,RefNum10};

        trailingAvgG = [];
        trailingAvgSqG = [];
        trailingAvgD = [];
        trailingAvgSqD = [];

        iteration = 0;

        % Loop over epochs.
        for epoch = 1:numEpochs
            % Reset and shuffle datastore.
            ref=rand(numCases,1)*numBatch;

            ZS=ZZ_SC(:,:,epoch+10,:);
            % Loop over mini-batches.
            for mbq=1:numBatch
                iteration = iteration + 1;

                disp([numRef, TypeID, iteration]);

                Z = ZS(:,1:(numRef+1),:,(ref<mbq)&(ref>mbq-1));


                [~,~,gradientsG,stateG] = ...
                    dlfeval(@modelLossG,netD,netG,gpuArray(Z));
                %gradientsG=extractdata(gradientsG);
                %stateG=extractdata(stateG);
                netG.State = stateG;
                [netG,trailingAvgG,trailingAvgSqG] = adamupdate(netG, gradientsG, ...
                    trailingAvgG, trailingAvgSqG, iteration, ...
                    learnRate*2, gradientDecayFactor, squaredGradientDecayFactor);

                %gpuDevice(2);
                [~,~,gradientsD,stateD] = ...
                    dlfeval(@modelLossDwG,netD,netG,gpuArray(Z));
                %gradientsD=extractdata(gradientsD);
                %stateD=extractdata(stateD);
                netD.State = stateD;
                [netD,trailingAvgD,trailingAvgSqD] = adamupdate(netD, gradientsD, ...
                    trailingAvgD, trailingAvgSqD, iteration, ...
                    learnRate, gradientDecayFactor, squaredGradientDecayFactor);
            end

            if (mod(epoch,ValidationFrequency)==0)
                DsRecord{TypeID,RefNum10,epoch/ValidationFrequency}=netD;
                GsRecord{TypeID,RefNum10,epoch/ValidationFrequency}=netG;
            end
        end
        Ds{TypeID,RefNum10}=netD;
        Gs{TypeID,RefNum10}=netG;
    end
end

%save('Step2_SC_GD.mat','Ds','DsRecord','Gs','GsRecord');
save('Step2_SC_GD_simp.mat','Ds','Gs');