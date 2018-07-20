%% Make Map for the whole image and colromap generation based on the needed structure!
%  @ Code modified by Shekoofeh Azizi on 20/10/2017 (UBC-RCL)

%%
clear all %#ok<CLALL>
close all
clc

%% Initializing: Define Parameters and Reading Data
path = 'E:\Feature Extraction\Philips Dataset\Extracted Features\';
log_dir = 'E:\tscRF_LSTM\Python\TeUS_RNN\TeUS_RNN\Datasets\logs\DeepNetArch3-Div\test_logs\';
filename = '2017_10_20_09_48_15_arch3_whole.mat';
% filename = '2017_10_19_09_55_17_arch2_whole.mat';
% filename = '2017_10_19_20_54_29_arch1_whole.mat';

% ExcelFileName : Name of Excel which contain our patients info
ExcelFileName = 'PatientsInfo_All.xlsx';

% Read filenames and Patient Info
[num,txt,raw] = xlsread(ExcelFileName);

PatientsInfo_FileName = txt(2:end,3);
PatientsInfo_FileName = cell2mat(PatientsInfo_FileName);

% Load probability maps
load([log_dir, filename]);
load('./Datasets/D_Whole_Labels.mat');

prob_estimates_test = test_predictions;
SampSize_test = infoCorew(:,10);
% Create structure containing filenames and corresponding probability maps
field1 = 'filename';  value1 = 'rf00000000000000';
field2 = 'probmap';   value2 = zeros(1,1);
s_temp = struct(field1,value1,field2,value2);

s = [];
noFiles = size(PatientsInfo_FileName,1);

for i=1:noFiles


    p_temp = prob_estimates_test(sum(SampSize_test(1:i-1))+ 1 : sum(SampSize_test(1:i)),1);
    
    filename = [path,'/features_wholeimage_limits_80_new/feature_limit_',PatientsInfo_FileName(i,:),'.mat'];
    load(filename);
    
    x_range = (x_lim_right - x_lim_left +0.5)*2;
    y_range = (y_lim_right - y_lim_left +0.5)*2;
    
    % For dataset 1-6th we don't have 
    if(x_range < 1 || y_range < 1)
        s_temp.filename = PatientsInfo_FileName(i,:);
        s = [s;s_temp];
        continue;
    end
    % Create structure containing filenames and corresponding probability maps
    field1 = 'filename';  value1 = 'rf00000000000000';
    field2 = 'probmap';   value2 = zeros(x_range,y_range);
    s_temp = struct(field1,value1,field2,value2);
    
    p_temp = reshape(p_temp,[y_range x_range]);
    probabilitymap = flip(p_temp);

    probabilitymap = imresize( probabilitymap, 'Scale', 0.5 );  %% Scale for 0.5 mm ROI s

    filename = PatientsInfo_FileName(i,:);
    s_temp.filename = filename;
    s_temp.probmap = probabilitymap;
    s = [s;s_temp];
    i
end


save RNN_Bmode_Wholemap.mat s
