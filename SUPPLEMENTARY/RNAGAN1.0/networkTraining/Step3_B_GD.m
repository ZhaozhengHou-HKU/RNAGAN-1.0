rng(3);

numGene=size(pathwayMatrix,2);
%numCases=numel(RefCaseWise);

numEpochs = 1000;
learnRate = 0.02;
gradientDecayFactor = 0.9;
squaredGradientDecayFactor = 0.999;
numBatch=50;
ValidationFrequency = 500;

DsRecord=cell(3,3,numEpochs/ValidationFrequency);
GsRecord=cell(3,3,numEpochs/ValidationFrequency);

for RefNum10=1:3

    numRef=10*RefNum10;
    for TypeID=1:3
        netD=Ds{TypeID,RefNum10};
        netG=Gs{TypeID,RefNum10};
        if (TypeID==3)
            netG=setLearnRateFactor(netG,"Pathways","Weights",0);
            netD=setLearnRateFactor(netD,"Pathways","Weights",0);
        end


        trailingAvgG = [];
        trailingAvgSqG = [];
        trailingAvgD = [];
        trailingAvgSqD = [];

        iteration = 0;

        % Loop over epochs.
        for epoch = 1:numEpochs

            Z=ZZ_B(:,1:(numRef+1),epoch,:);
            % Loop over mini-batches.
            iteration = iteration + 1;

            disp([numRef, TypeID, iteration]);

            [~,~,gradientsG,stateG] = ...
                dlfeval(@modelLossG,netD,netG,gpuArray(Z));
            %gradientsG=extractdata(gradientsG);
            %stateG=extractdata(stateG);
            netG.State = stateG;
            [netG,trailingAvgG,trailingAvgSqG] = adamupdate(netG, gradientsG, ...
                trailingAvgG, trailingAvgSqG, iteration, ...
                learnRate, gradientDecayFactor, squaredGradientDecayFactor);

            %gpuDevice(2);
            [~,~,gradientsD,stateD] = ...
                dlfeval(@modelLossDwG,netD,netG,gpuArray(Z));
            %gradientsD=extractdata(gradientsD);
            %stateD=extractdata(stateD);
            netD.State = stateD;
            [netD,trailingAvgD,trailingAvgSqD] = adamupdate(netD, gradientsD, ...
                trailingAvgD, trailingAvgSqD, iteration, ...
                learnRate, gradientDecayFactor, squaredGradientDecayFactor);

            if (mod(epoch,ValidationFrequency)==0)
                DsRecord{TypeID,RefNum10,epoch/ValidationFrequency}=netD;
                GsRecord{TypeID,RefNum10,epoch/ValidationFrequency}=netG;

                %gpuDevice(2);

            end
        end
        Ds{TypeID,RefNum10}=netD;
        Gs{TypeID,RefNum10}=netG;
    end
end

%save('Step3_B_GD_Same.mat','Ds','Gs');
save('Step3_B_GD_simp.mat','Ds','DsRecord','Gs','GsRecord');