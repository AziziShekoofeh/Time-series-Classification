function [filter_Core] = makeFilter(Y_Core,filtering,value)
%% Exclude data based o different criteria
%   nofilter : include all of the data
%   basic: exclude cores with more than 3mm distance to boundary and cores
%   have mismatche between axial and sagittal pathology
%   mrbased: basic filter + MR grade based exclusion
%   gsbased: basic filter + GS based exclusion
%   binary: sepration of cancer and benign.

%   Inputs: Y_ROI ,Y_Core : Labels and charectristics of each ROI/Core
%           filtering: string, based on following filters, i.e. 'basic'
%           value: default-0 for basic, nofilter,
%                  mrbased-1(low),2(medium),3(high)
%                  gsbased-0,6,71(GS 3+4),72(GS 4+3),8,9
%                  binary: 0 Benign, 1 Cancerous
%
%  Patient Info:
%  Column 1: Label
%  Column 2: Tumor in Core length
%  Column 3: MR Label
%  Column 4: Axial and Sagittal Match (1: Match 0: Mismatch)
%  Column 5: Gleason Score
%  Column 6: Distance to Boundary (mm)
%  Column 7: MR greatest size
%  Column 8: Sagittal GS
%  Column 9: Sample number
%  Column 10: ROI number/ Column 10: Sample Size

%   @ Code composed by Shekoofeh Azizi on 24/11/2015 (UBC-RCL)
%   @ Code modified by Shekoofeh Azizi on 01/06/2017 (UBC-RCL)

%%
if nargin < 3
    assert(~strcmp(filtering,'D2Bmrbased'),'Invalid value for MR grade!');
    value = 0;
end

% set filtering conditions
switch filtering
    case 'nofilter'
        %         filter_ROI = 1:length(Y_ROI) ;
        filter_Core = 1:length(Y_Core) ;
    case 'D2'
        %         filter_ROI = 1:length(Y_ROI) ;
        filter_Core = 1:length(Y_Core) ;
    case 'binary'
        %         filter_ROI = find(Y_ROI(:,1)== value);
        filter_Core = find(Y_Core(:,1)== value);
    case 'D2B'
        % less than 3mm and mismatch
        %         filter_ROI = find(Y_ROI(:,8) >= 2.99 & Y_ROI(:,6)== 1 ) ;
        filter_Core = find(Y_Core(:,6) >= 2.99 & Y_Core(:,4)== 1 );
    case 'D3B'
        % less than 3mm and mismatch
        %         filter_ROI = find(Y_ROI(:,8) >= 2.99 & Y_ROI(:,6)== 0 ) ;
        filter_Core = find(Y_Core(:,6) >= 2.99 & Y_Core(:,4)== 0 );
    case 'D2A'
        % less than 3mm and mismatch
        %         filter_ROI = find(Y_ROI(:,8) >= 2.99 ) ;
        filter_Core = find(Y_Core(:,6) >= 2.99 );
    case 'D3'
        % Just less than 3mm
        %         filter_ROI = find(Y_ROI(:,8) < 3.00 ) ;
        filter_Core = find(Y_Core(:,6) < 3.00 );
    case 'D2C'
        % Include match cores
        %         filter_ROI = find(Y_ROI(:,6)== 1) ;
        filter_Core = find(Y_Core(:,4)== 1);
    case 'D2Cmrlen'
        % Include match cores
        %         filter_ROI = find(Y_ROI(:,6)== 1 & Y_ROI(:,13) >= 2) ;
        filter_Core = find(Y_Core(:,4)== 1 & Y_Core(:,13) >= 2);
    case 'D2M'
        % Include just mis-match cores
        %         filter_ROI = find(Y_ROI(:,6)== 0) ;
        filter_Core = find(Y_Core(:,4)== 0);
    case 'D2mrbased'
        % less than 3mm and mismatch + MR level filtering
        %         filter_ROI = find(Y_ROI(:,5)== value) ;
        filter_Core = find(Y_Core(:,3)==value);
    case 'D2Bmrbased'
        % less than 3mm and mismatch + MR level filtering
        %         filter_ROI = find(Y_ROI(:,8) >= 2.99 & Y_ROI(:,6)== 1 & Y_ROI(:,5)== value) ;
        filter_Core = find(Y_Core(:,6) >= 2.99 & Y_Core(:,4)== 1 & Y_Core(:,3)==value);
    case 'D2Amrbased'
        % less than 3mm and mismatch + MR level filtering
        %         filter_ROI = find(Y_ROI(:,8) >= 2.99 & Y_ROI(:,5)== value) ;
        filter_Core = find(Y_Core(:,6) >= 2.99 & Y_Core(:,3)==value);
    case 'D2Cmrbased'
        % less than 3mm and mismatch + MR level filtering
        %         filter_ROI = find(Y_ROI(:,6)== 1 & Y_ROI(:,5)== value) ;
        filter_Core = find(Y_Core(:,4)== 1 & Y_Core(:,3)==value);
    case 'gsbased'
        % less than 3mm and mismatch + Gleason filtering
        %         filter_ROI = find(Y_ROI(:,8) >= 2.99 & Y_ROI(:,6)== 1 & Y_ROI(:,7)== value) ;
        filter_Core = find(Y_Core(:,6) >= 2.99 & Y_Core(:,4)== 1 & Y_Core(:,5)== value);
    case 'D2Cgsbased'
        % less than 3mm and mismatch + Gleason filtering
        %         filter_ROI = find(Y_ROI(:,8) >= 2.99 & Y_ROI(:,6)== 1 & Y_ROI(:,7)== value) ;
        filter_Core = find(Y_Core(:,4)== 1 & Y_Core(:,5)== value);
    case 'Lenbased'
        filter_Core = find(Y_Core(:,4) >= value & Y_Core(:,4)== 1);
end

end