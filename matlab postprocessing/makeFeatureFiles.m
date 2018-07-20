function [X_bmode, X_rf, infoROI, infoCore, PatientsInfo_FileName] = makeFeatureFiles(path_1, path_2)

%% Generate Feature Design Matrix for all of the B-mode and RF data 
%  (Philips Dataset Including 255 Test 80 ROIs)

% INPUT
%   Saving Path: path_1
%   Feature Path: Path_2 

% OUTPUT
%   X_bmode : Bmode Features
%   X_rf    : RF Features
%   S_info    : Patient information and labels

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

%   @ Code composed by Shekoofeh Azizi on 22/08/2016 (UBC-RCL)
%   @ Code modified by Shekoofeh Azizi on 19/05/2017 (UBC-RCL)

%% Read our Excel in format of table, contain the info of patients
ExcelFileName = [path_1,'PatientsInfo_All.xlsx'];
[num,txt,~] = xlsread(ExcelFileName);
PatientsInfo_FileName = txt(2:end,3);
PatientsInfo_FileName = cell2mat(PatientsInfo_FileName);
PatientsInfo = num(1:end,[3, 4, 5, 6, 9, 19, 20, 15, 22]);

%% Ceating Matrix of Features (X) for the RF
X = [];
infoROI = [];
feature = [];
size_samples = [];

for i = 1 : size(PatientsInfo_FileName,1)
    filename = [path_2,'./features_rf_tsc_ROI_80/feature_',PatientsInfo_FileName(i,:),'.mat'];
    load(filename);
    X = [X; feature];
    size_samples = [size_samples; size(feature,1)];
    ROI_num = 1:size(feature,1);
    infoROI = [infoROI; [repmat([PatientsInfo(i,:)],[size(feature,1),1]),ROI_num']];
end
X_rf = X;
infoCore = [PatientsInfo,size_samples];

%% Ceating Matrix of Features (X) for the Bmode
X = [];
feature = [];
for i = 1 : size(PatientsInfo_FileName,1)
    filename = [path_2,'./features_bmode_tsc_ROI_80/feature_',PatientsInfo_FileName(i,:),'.mat'];
    load(filename);
    X = [X;feature];
end
X_bmode = X;

removeIDX = (any(isnan(X_rf),2));
X_rf(removeIDX,:)= 0 ;

end