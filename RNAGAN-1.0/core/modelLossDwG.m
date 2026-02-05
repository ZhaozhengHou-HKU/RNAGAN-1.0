function [lossD,scoreD,gradientsD,stateD] = ...
    modelLossDwG(netD,netG,Z)

temp=find(rand(size(Z,4),1)<0.1);
if (numel(temp)>0)
    GZ = forward(netG,Z(:,2:end,:,temp));
end

% Calculate the predictions for real data with the discriminator network.
n=size(Z,4)+1;
Z=repmat(Z,[1,1,1,2]);
Z(:,1,1,n:end)=Z(:,1,1,[((n+1):end),end-1]);
if (numel(temp)>0)
    Z(:,:,:,end+(1:numel(temp)))=GZ;
end
[Y,stateD] = forward(netD,Z);
%Y=Y(:);
Y(n:end)=1-Y(n:end);

scoreD=mean(Y,"all");
lossD=-log(scoreD);

gradientsD = dlgradient(lossD,netD.Learnables);
end