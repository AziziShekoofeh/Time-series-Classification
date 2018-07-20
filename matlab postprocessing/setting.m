%   Script to set path for the training logs
%   @ Code composed by Shekoofeh Azizi on 18/10/2017 (UBC-RCL)

function [subFiles, opt_param_log_name, opt_param_log_value, learn_log] = setting(model_type)
% get root of current file
log_dir = 'E:\tscRF_LSTM\Python\TeUS_RNN\TeUS_RNN\Datasets\logs\';
log_dir_sub = '/'; % or /opt/ or /bmode/ or /

% RF model id: Optimum Params
lstm_opt_model_id = '2017_09_10_21_41_54_arch1.csv';
gru_opt_model_id = '2017_09_11_18_58_33_arch2.csv';
rnn_opt_model_id = '2017_09_07_23_03_26_arch3.csv';

switch model_type
    case 'lstm'
        sub_dir = 'DeepNetArch1-Div';
        % Add Path
        root = [log_dir, sub_dir, log_dir_sub];
        addpath([root '/train_logs']);
        addpath([root '/csv_logs']);
        [opt_param_log_name, opt_param_log_value]  = importfilecsv(lstm_opt_model_id);
        learn_log = importdata([lstm_opt_model_id(1:25),'.log']);
        
    case 'gru'
        sub_dir = 'DeepNetArch2-Div';
        % Add Path
        root = [log_dir, sub_dir, log_dir_sub];
        addpath([root '/train_logs']);
        addpath([root '/csv_logs']);
        [opt_param_log_name, opt_param_log_value]  = importfilecsv(lstm_opt_model_id);
        learn_log = importdata([gru_opt_model_id(1:25),'.log']);
    case 'rnn'
        sub_dir = 'DeepNetArch3-Div';
        % Add Path
        root = [log_dir, sub_dir, log_dir_sub];
        addpath([root '/train_logs']);
        addpath([root '/csv_logs']);
        [opt_param_log_name, opt_param_log_value]  = importfilecsv(lstm_opt_model_id);
        learn_log = importdata([rnn_opt_model_id(1:25),'.log']);
end




% Get a list of all files and folders in this folder.
files = dir([log_dir, sub_dir, '\', '\csv_logs\']);
% Get a logical vector that tells which is a directory.
dirFlags = [files(:).isdir];
% Extract only those that are directories.
subFiles = {files(~dirFlags).name}';
% Removing current and previous directory
% subFolders(ismember(subFolders,{'.','..'})) = [];

end