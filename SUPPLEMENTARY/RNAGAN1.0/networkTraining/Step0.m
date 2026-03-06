rng(0);

Ds=cell(3);
Gs=cell(3);


for b=1:3
    a=1;%NP
    PM=pathwayMatrix([],:);
    Ds{a,b}=DiscriminatorNetwork0(PM,10*b);
    Gs{a,b}=GeneratorNetwork(PM,10*b);
    a=2;%PP
    PM=pathwayMatrix;
    Ds{a,b}=DiscriminatorNetwork(PM,10*b);
    Gs{a,b}=GeneratorNetwork(PM,10*b);
    a=3;%LP
    PM=true(size(pathwayMatrix));
    Ds{a,b}=DiscriminatorNetwork(PM,10*b);
    Gs{a,b}=GeneratorNetwork(PM,10*b);
end

save('step0.mat','Ds','Gs');