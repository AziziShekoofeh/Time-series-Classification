function [selected_idx_train, selected_idx_test] = dataSelection(D,L,TCL_limit,MTL_limit)

%% Select data for Validation Test and Train based on th etumor size
%   @ Code composed by Shekoofeh Azizi on 23/08/2016 (UBC-RCL)
%   @ Code modified by Shekoofeh Azizi on 19/05/2017 (UBC-RCL)

%% 1- Train + Validation Data

noROI = 80;
trainPer = 1.0;

% Select cancerous large cores
L_ca = L(L(:,1) == 1,:);
s = RandStream('mt19937ar','Seed',0);
selected_ca_train = find( L_ca(:,2) >= TCL_limit & L_ca(:,7) >= MTL_limit & L_ca(:,4) == 1);
selected_ca_train = randperm(s,length(selected_ca_train),length(selected_ca_train));
selected_ca_train = L_ca(selected_ca_train,9);
%D_ca_train = D_ca(ExpandPSamp(selected_ca_train,noROI),:);

% Select benign cores
L_be = L(L(:,1) == 0 & L(:,4) == 1,:);

% Fix a seed to generate a reproducible results
s = RandStream('mt19937ar','Seed',10);
selected_be_train = randperm(s,length(L_be),length(selected_ca_train)); % Equal number of cancerous and benign
selected_be_train = L_be(selected_be_train,9);
% selected_be_train = ExpandPSamp(selected_be_train,noROI);

%% Selected index

selected_idx_train = [selected_be_train; selected_ca_train];
selected_idx_test = find(~ismember(L(:,9),selected_idx_train));

if(intersect(selected_idx_train,selected_idx_test))
    warning('Error in dataselection');
    display(intersect(selected_idx_train,selected_idx_test))
end
end
