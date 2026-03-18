function [lossG,scoreG,gradientsG,stateG] = ...
    modelLossG(netD,netG,Z)
% compute the loss for Generator

% Calculate the predictions for real data with the discriminator network.
[GZ,stateG] = forward(netG,Z(:,2:end,:,:));

Y = forward(netD,(GZ));
%Y=Y(:);

scoreG=mean(Y,'all');
lossG=-log(scoreG);

gradientsG = dlgradient(lossG,netG.Learnables);
end