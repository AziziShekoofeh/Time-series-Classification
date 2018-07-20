%% Comparison of Bmode and RF data
% @ Code composed by Shekoofeh Azizi on 22/05/2017 (UBC-RCL)
% @ Code modified by Shekoofeh Azizi on 01/06/2017 (UBC-RCL)
% @ Code modified by Shekoofeh Azizi on 20/10/2017 (UBC-RCL)

%%
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

%%

clear all %#ok<CLALL>
close all
clc

noROI = 80;
filtering = 'nofilter';
value = 2;

%% Loading RF model and results
log_dir = 'E:\tscRF_LSTM\Python\TeUS_RNN\TeUS_RNN\Datasets\logs\DeepNetArch1-Div\test_logs\';

% Bmode model id
% filename = '2017_10_20_09_48_15_arch3.mat'; % Threshold .5  Wholemap .8
% filename = '2017_10_19_09_55_17_arch2.mat'; % Threshold .4
% filename = '2017_10_19_20_54_29_arch1.mat'; % Threshold .5  Wholemap .8

% RF model id
% filename = '2017_10_05_14_48_49_arch3.mat'; % Threshold .5  Wholemap .8
% filename = '2017_10_05_11_28_48_arch2.mat'; % Threshold .5
% filename = '2017_10_05_11_29_04_arch1.mat'; % Threshold .4  Wholemap .8
filename = '2017_10_08_19_38_47_arch1.mat';

load([log_dir, filename]);
load('.\Datasets\D_Fixed')

L_TEST = Lf_test(1:size(Lf_test)/2,:);
noFiles_test = size(L_TEST,1)/noROI;
estimatedProb = test_predictions;

% Find the optimm threshold using ROC curve
[X_ROC,Y_ROC,T,~,OPTROCPT] = perfcurve(L_TEST(:,1),estimatedProb,1);
Threshold = T((X_ROC==OPTROCPT(1))&(Y_ROC==OPTROCPT(2))); 
% Threshold = 0.5;
predictedL = (estimatedProb>=Threshold);

CancerPercentage = [];
CancerEstimate = [];
L_Core = [];
for i = 1 : noFiles_test 
    predict_label = predictedL((i-1)*noROI+1:i*noROI,1);
    cancer_estimate = estimatedProb((i-1)*noROI+1:i*noROI,1);
    CancerPercentage(i)=100*length(find(predict_label==1))/noROI; %#ok<SAGROW>
    CancerEstimate(i)=100*sum(cancer_estimate(:,1))/noROI; %#ok<SAGROW>
    L_Core(i,:) = L_TEST((i-1)*noROI+1,:); %#ok<SAGROW>
end
CancerPercentage = CancerPercentage';
CancerEstimate = CancerEstimate';
results =  CancerPercentage;

%% Results Evalution
hold on
filter = makeFilter(L_Core,filtering,value);
[X_ROC,Y_ROC,T,AUC_Core,OPTROCPT] = perfcurve(L_Core(filter,1),double(results(filter,:)),1);
plot(X_ROC,Y_ROC,'Color','b','LineWidth',1.5,'DisplayName','RF data')
xlabel('False positive rate (1-Specificity)'); 
ylabel('True positive rate (Specificity)')

noFiles_filter = size(L_Core(filter,1),1);
cp = CancerPercentage(filter,1);
sen = OPTROCPT(2);
spe = 1-OPTROCPT(1);
Threshold_filter = T((X_ROC==OPTROCPT(1))&(Y_ROC==OPTROCPT(2)));
acc = sum(L_Core(filter,1) == (cp >= Threshold_filter))/size(cp,1);
display('RF Results')
fprintf('Accuracy: %d   AUC: %d\n', acc, AUC_Core);
fprintf('Sensitivity: %d\n', sen);
fprintf('Specificity: %d\n', spe);
predictedL= (cp >= Threshold_filter); 
CancerP = results(filter,1);

%%
% Plot AUC vs. MR length for binary classification
MRsize = L_Core(filter,7);
realGrade = L_Core(filter,1);
[~,AUC_Predicted] = plotMRvsAccuracy(MRsize,predictedL,realGrade,CancerP);


%% Binary classification + MR grading
MRgrade = L_Core(filter,3);
[~,CancerPercentageCombined] = gradeCombination(predictedL,MRgrade,CancerP);
[~,~,~,AUC_Core,~] = perfcurve(L_Core(filter,1),CancerPercentageCombined,1);
fprintf('AUC Combined: %d\n', AUC_Core);

