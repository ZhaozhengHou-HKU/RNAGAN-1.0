function [P,G] = visualize_2order_features...
    (gradients,R2,markerNames,selectedMarkers)
%VISUALIZE_2ORDER_FEATURES   Generate plot the 2nd order features
%   author: Zhaozheng Hou (George)
%
% [pseudo,pseudoNet] = visualize_2order_features(gradients,R2,
%       markerNames,selectedMarkers)
% parameter:
%   - gradients: gradient matrix of features
%   - R2: partial R2 values fo the features
%   - markerNames: (optional) marker names, feature order by default.
%   - selectedMarkers: (optional) features to show, show all by default.
% output:
%   - P: plot object
%   - G: graph object

%% validate
validateattributes(gradients,{'numeric'},{'square'});
validateattributes(R2,{'numeric'},{'square','size',size(gradients)});
if (nargin<3)
    markerNames=1:(size(gradients,1));
end
markerNames=string(markerNames);

if (nargin<4)
    selectedMarkers=1:(size(gradients,1));
else
    if (isstring(selectedMarkers))
        selectedMarkers=...
            arrayfun(@(str)find(markerNames==str),selectedMarkers);
    else
        if (max(selectedMarkers)<2)
            selectedMarkers=find(selectedMarkers>0);
        end
        validateattributes(selectedMarkers,{'numeric'},{'<=',size(gradients,1)});
    end
end

%% process
n=numel(selectedMarkers);
gradients=gradients(selectedMarkers,selectedMarkers);
R2=R2(selectedMarkers,selectedMarkers);
if (numel(markerNames)>numel(selectedMarkers))
    markerNames=markerNames(selectedMarkers);
end

cutoff=max(0.05,quantile(R2,0.95,"all"));
dgg=diag(gradients)>0;
for ind=1:n
    R2(ind,ind)=0;
    R2(ind,dgg(ind)&dgg&(gradients(:,ind)<0))=0;
    R2(ind,~dgg(ind)&~dgg&(gradients(:,ind)>0))=0;
end

G=graph(R2>=cutoff,"lower");
G.Edges.Weight=...
    arrayfun(@(id) R2(G.Edges.EndNodes(id,1),G.Edges.EndNodes(id,2)),...
    (1:size(G.Edges,1))');
G.Edges.GradientDir=...
    arrayfun(@(id) gradients(G.Edges.EndNodes(id,1),G.Edges.EndNodes(id,2)),...
    (1:size(G.Edges,1))');
values=quantile(diag(gradients),[0,1]);
G.Edges.GradientDir(G.Edges.GradientDir<0)=values(1);
G.Edges.GradientDir(G.Edges.GradientDir>0)=values(2);
G.Nodes.Name=markerNames(:);

figure();
P=plot(G);
layout(P,'force','WeightEffect','inverse','Iterations',512);
P.NodeCData=diag(gradients);
P.EdgeCData=G.Edges.GradientDir;
colormap("turbo");
P.MarkerSize=7;
P.LineWidth=2;

end