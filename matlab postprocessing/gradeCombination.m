function [combinedGrade,CancerPercentageCombined] = gradeCombination(Predicted_Label,MRgrade,CancerPercentage)

% Grade combination for benign/cancerous classifier
% Predicted_Label: from cancer detection approach, 1 means cancer 0 means non-cancerous
% MRgrade: from MRI, 1 = low, 2 = moderate, 3 = high


%   @ Code composed by Shekoofeh Azizi on 01/02/2016 (UBC-RCL)
%   @ Code modified by Shekoofeh Azizi on 29/05/2017 (UBC-RCL)

noCores = size(MRgrade,1);
combinedGrade = Predicted_Label;
CancerPercentageCombined = CancerPercentage;

for i =1 : noCores     
%    if(MRgrade(i,1) == 3 && Predicted_Label(i,1)==0 && CancerPercentage(i,1)~=0 )
    if(MRgrade(i,1) == 3 && Predicted_Label(i,1)==0 )
        combinedGrade(i,1)= 1;
        CancerPercentageCombined(i,1) = 100;
    end

    if(MRgrade(i,1) == 1 && Predicted_Label(i,1)==1)
        combinedGrade(i,1)= 0;
        CancerPercentageCombined(i,1) = 0;
    end
end

end