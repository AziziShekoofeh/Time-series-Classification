function [X_bmode, X_rf, infoCore, infoROI] = makeFeatureFiles_wholeimage(path_1, path_2)

%% Generate Feature Design Matrix for the B-mode and Rf data in Whole Image 
% They are unlabeled data
%  (Philips Dataset Including 255 Test 80 ROIs)

%INPUT
%   Saving Path:  path_1
%   Feature Path: Path_2 

% OUTPUT
%   X_bmode : Bmode Features
%   X_rf    : RF Features

%   @ Code composed by Shekoofeh Azizi on 29/11/2015 (UBC-RCL)
%   @ Code modified by Shekoofeh Azizi on 19/05/2017 (UBC-RCL)
%   @ Code modified by Shekoofeh Azizi on 20/10/2017 (UBC-RCL)

%% Read our Excel in format of table, contain the info of patients
ExcelFileName = [path_1,'PatientsInfo_All.xlsx'];
[num,txt,~] = xlsread(ExcelFileName);
PatientsInfo_FileName = txt(2:end,3);
PatientsInfo_FileName = cell2mat(PatientsInfo_FileName);
PatientsInfo = num(1:end,[3, 4, 5, 6, 9, 19, 20, 15, 22]);

%% Ceating Matrix of Features (X) for the Bmode
X = [];
S = [];
feature = [];
infoROI = [];
size_samples = [];

for i = 1 : size(PatientsInfo_FileName,1)
    filename = [path_2,'./features_bmode_tsc_wholeimage_80/feature_',PatientsInfo_FileName(i,:),'.mat'];
    load(filename);
    X = [X; feature]; %#ok<AGROW>
    infoROI = [infoROI; repmat(PatientsInfo(i,:),[size(feature,1),1])]; %#ok<AGROW>
    size_samples = [size_samples; size(feature,1)]; %#ok<AGROW>
end
X_bmode = X;
infoCore = [PatientsInfo,size_samples];

%% Ceating Matrix of Features (X) for the RF
X = [];
feature = [];
for i = 1 : size(PatientsInfo_FileName,1)
    filename = [path_2,'./features_rf_tsc_wholeimage_80/feature_',PatientsInfo_FileName(i,:),'.mat'];
    load(filename);
    X = [X; feature]; %#ok<AGROW>
end
X_rf = X;

end