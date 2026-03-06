function [netGeneID,listGeneID,matched] = match_gene_list(geneList, format)
%MATCH_GENE_LIST    match a given gene list to the list for RNAGAN network.
%   Only the unique matches would remain.
%   author: Zhaozheng Hou (George)
%
% Example
%   -------
%       network(netGeneID)=data(listGeneID);
%       network(netGeneID)=data(matched);
%       data(listGeneID)=network(netGeneID);
%
% [netGeneID,listGeneID,matched] = match_gene_list(geneList, format)
% parameter:
%   - geneList: the list of genes (strings or numbers)
%   - format: (optional) the format of gene list: "ENSG"/"Entrez"/"symbol"
% output:
%   - netGeneID: the IDs of matched genes in net input layer
%   - listGeneID: the IDs of matched genes in gene list
%   - matched: bool array telling which gene on the list got unique match

%% validate
validateattributes(geneList,{'numeric','char','string'},{'vector'});
if (nargin==1)
    format="default";
else
    validateattributes(format,{'char','string'},"scalartext");
end

%% process
lists=load('core\Genes_n_Pathways.mat',...
    'ENSG','ENSG_ID','Entrez_ID','symbol');

matching=-1;
if (isnumeric(geneList))
    geneList=geneList(:)';

    % ENSG only
    if (contains(format,"ENSG",IgnoreCase=true))
        matching=(lists.ENSG_ID==geneList);
        matching(sum(matching,2)>1,:)=false;
        if (sum(matching)==0)
            error("Cannot find any matched gene.");
        end
    end
    % Entrez_ID only
    if (contains(format,"Entrez",IgnoreCase=true))
        matching=(lists.Entrez_ID==geneList);
        matching(:,sum(matching,1)>1)=false;
        matching(sum(matching,2)>1,:)=false;
        if (sum(matching)==0)
            error("Cannot find any matched gene.");
        end
    end

    % check both ENSG and Entrez by default
    if (matching==-1)
        geneList=floor(geneList(:))';

        matching=(lists.ENSG_ID==geneList);
        matching(sum(matching,2)>1,:)=false;
        matched=sum(matching);
        matching=(lists.Entrez_ID==geneList);
        matching(:,sum(matching,1)>1)=false;
        matching(sum(matching,2)>1,:)=false;
        if (sum(matching)>matched)
            % should be Entrez_ID
            matched=sum(matching);
            if (matched==0)
                error("Cannot find any matched gene.");
            end
            disp("The format is automatically idetified as Entrez ID.");
        else
            % should be ENSG_ID
            if (matched==0)
                error("Cannot find any matched gene.");
            end
            disp("The format is automatically idetified as ENSG ID.");
            matching=(lists.ENSG_ID==geneList);
            matching(sum(matching,2)>1,:)=false;
        end
    end
else
    geneList=string(geneList);
    for ind=1:numel(geneList)
        if contains(geneList(ind),'.')
            temp=split(geneList(ind),'.');
            geneList(ind)=temp(1);
        end
    end
    geneList=geneList(:)';

    % ENSG only
    if (contains(format,"ENSG",IgnoreCase=true))
        matching=(lists.ENSG==geneList);
        matching(sum(matching,2)>1,:)=false;
        if (sum(matching)==0)
            error("Cannot find any matched gene.");
        end
    end
    % symbol only
    if (contains(format,"symbol",IgnoreCase=true))
        matching=(lists.symbol==geneList);
        matching(:,sum(matching,1)>1)=false;
        matching(sum(matching,2)>1,:)=false;
        if (sum(matching)==0)
            error("Cannot find any matched gene.");
        end
    end

    % check both ENSG and symbol by default
    if (matching==-1)
        matching=(lists.ENSG==geneList);
        matching(sum(matching,2)>1,:)=false;
        matched=sum(matching);
        matching=(lists.symbol==geneList);
        matching(:,sum(matching,1)>1)=false;
        matching(sum(matching,2)>1,:)=false;
        if (sum(matching)>matched)
            % should be symbol
            matched=sum(matching);
            if (matched==0)
                error("Cannot find any matched gene.");
            end
            disp("The format is automatically idetified as gene symbol.");
        else
            % should be ENSG
            if (matched==0)
                error("Cannot find any matched gene.");
            end
            disp("The format is automatically idetified as ENSG.");
            matching=(lists.ENSG==geneList);
            matching(sum(matching,2)>1,:)=false;
        end
    end

end
matched=any(matching,1);
disp(num2str([sum(matched),sum(matched)/numel(geneList)*100],...
    "%u matched genes (%0.2f%% of the gene list)"));
[netGeneID,listGeneID]=find(matching);
end