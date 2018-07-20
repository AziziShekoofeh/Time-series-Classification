%% makeData: Data division for large cores in LSTM impelemtaion
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Patient Info:
% Column 1: Label
% Column 2: Tumor in Core length   
% Column 3: MR Label
% Column 4: Axial and Sagittal Match (1: Match 0: Mismatch)
% Column 5: Gleason Score
% Column 6: Distance to Boundary (mm)
% Column 7: MR greatest size
% Column 8: Sagittal GS
% Column 9: Sample number
% Column 10: ROI number/ Column 10: Sample Size

% @ Code composed by Shekoofeh Azizi on 24/10/2016 (UBC-RCL)
% @ Code modified by Shekoofeh Azizi on 23/12/2016 (UBC-RCL)
% @ Code modified by Shekoofeh Azizi on 19/05/2017 (UBC-RCL)
% @ Code modified by Shekoofeh Azizi on 15/08/2017 (UBC-RCL)

%%
clc
close all
clear all %#ok<CLALL>

%% Initialization
TCL_limit = 4.00; % more than 25% be cancerous  
MTL_limit = 0.0;  % Large cores 

path_1 = 'E:\tscRF_LSTM\Python\TeUS_RNN\TeUS_RNN\matlab postprocessing\Data preparation for RNN based methods\';  % Saving Path: path_1
path_2 = 'E:\Feature Extraction\Philips Dataset\Extracted Features\';    % Feature Path: Path_2 

%% Reading Data
% 1- Reading data from the sliding ROI
[Xs_bmode, Xs_rf, infoROIs, infoCores, ~] = makeFeatureFiles_Sliding(path_1, path_2);

% 2- Reading data from the fixed ROI
[Xf_bmode, Xf_rf, infoROIf, infoCoref, ~] = makeFeatureFiles(path_1, path_2);

% 3- Make the whole image feature files
[Xw_bmode, Xw_rf, infoCorew, infoROIw] = makeFeatureFiles_wholeimage(path_1, path_2);

%% Partitioning based on the tumore size
% Select large tumor size for train+validation
% Then select randomly between them for train(0.8) and validation(0.2)
% Equal number of benign cores selected randomly for each set
[selected_idx_train, selected_idx_test] = dataSelection(Xf_bmode,infoCoref,TCL_limit,MTL_limit);


%% Save D_Fixed
idx_train = find(ismember(infoROIf(:,9),selected_idx_train));
idx_test  = find(ismember(infoROIf(:,9),selected_idx_test));
Df_train  = [Xf_bmode(idx_train,:);Xf_rf(idx_train,:)];  
Lf_train  = [infoROIf(idx_train,:);infoROIf(idx_train,:)];
Df_test   = [Xf_bmode(idx_test,:);Xf_rf(idx_test,:)]; 
Lf_test   = [infoROIf(idx_test,:);infoROIf(idx_test,:)];
save([path_1,'Datasets\D_Fixed.mat'],'Df_train','Lf_train','Df_test','Lf_test')
 
%% Save D_Sliding
idx_train = find(ismember(infoROIs(:,9),selected_idx_train));
idx_test  = find(ismember(infoROIs(:,9),selected_idx_test));
Ds_train = [Xs_bmode(idx_train,:);Xs_rf(idx_train,:)];  
Ls_train = [infoROIs(idx_train,:);infoROIs(idx_train,:)];
Ds_test = [Xs_bmode(idx_test,:);Xs_rf(idx_test,:)]; 
Ls_test = [infoROIs(idx_test,:);infoROIs(idx_test,:)];
save([path_1,'Datasets\D_Sliding.mat'],'Ds_train','Ls_train','Ds_test','Ls_test')


%% Save D_Whole
Dw_bmode = Xw_bmode';
Dw_rf =Xw_rf';
save([path_1,'Datasets\D_Whole_Bmode.mat'],'Dw_bmode','-v7.3')
save([path_1,'Datasets\D_Whole_RF.mat'],'Dw_rf','-v7.3')
save([path_1,'Datasets\D_Whole_Labels.mat'],'infoCorew')