function [lossD,scoreD,gradientsD,stateD] = ...
    modelLossD(Dnet,Z)

% Calculate the predictions for real data with the discriminator network.
n=size(Z,4)+1;
Z=repmat(Z,[1,1,1,2]);
Z(:,1,1,n:end)=Z(:,1,1,[((n+1):end),end-1]);

[Y,stateD] = forward(Dnet,Z(:,:,:,:));
%Y=Y(:);
Y(n:end)=1-Y(n:end);

scoreD=mean(Y,"all");
lossD=-mean(log(Y));

gradientsD = gradient(lossD,Dnet.Learnables);
end