%   Plot Learning curves for the selected models
%   @ Code composed by Shekoofeh Azizi on 18/10/2017 (UBC-RCL)
%%
clc
clear all %#ok<CLALL>
close all

%% Initialization
% Se the model ids of the selected optimum models

% Bmode model id: Opt
% lstm_opt_model_id = '2017_10_19_20_54_29_arch1.csv';
% gru_opt_model_id = '2017_10_19_09_55_17_arch2.csv';
% rnn_opt_model_id = '2017_10_20_09_48_15_arch3.csv';

% RF model id: Opt
lstm_opt_model_id = '2017_10_05_11_29_04_arch1.csv';
gru_opt_model_id = '2017_10_05_11_28_48_arch2.csv';
rnn_opt_model_id = '2017_10_05_14_48_49_arch3.csv';

% Path setting
log_dir = 'E:\tscRF_LSTM\Python\TeUS_RNN\TeUS_RNN\Datasets\logs\';
log_dir_sub = '\'; 
sub_dir = 'DeepNetArch2-Div';
root = [log_dir, sub_dir, log_dir_sub];
addpath([root '/train_logs']);
addpath([root '/csv_logs']);

%%
curve_type = {'loss';'acc';'val_loss';'val_acc'};
curve_names = {'Train Loss';'Train Accuracy';'Validation Loss';'Validation Accuracy'};
linestyle = {'-'; '-.'; '-'; '-.' };
color = {[1 0.27 0.27] ; [1 0.27 0.27]; [0 0.8 0.4]; [0 0.8 0.4]};
noCurves = size(curve_type,1);

fig = figure;
left_color = [0 0 0];
right_color = [0 0 0];
set(fig,'defaultAxesColorOrder',[left_color; right_color]);
% Create axes
ax = axes('Parent',fig);
set(ax,'FontName','Times','FontSize',14,'GridColor',...
    [0.247058823529412 0.247058823529412 0.247058823529412],'GridLineStyle',':',...
    'LineStyleOrderIndex',3,'XGrid','on','YGrid','on');
for i = 1 : noCurves
    if(i<3)
        yyaxis left
        ylabel('Loss','FontName','Times','Interpreter','latex');
    else
        yyaxis right
        ylabel('Accuracy/AUC','FontName','Times','Interpreter','latex');
    end
    net_name = gru_opt_model_id(1:25);
    [param_log_name, param_log_value]  = importfilecsv([net_name, '.csv']);
    learn_log = importdata([net_name, '.log']);
    diagram_type_train = curve_type{i,1};
    [~, param_loc] = intersect(learn_log.textdata,cellstr(diagram_type_train));
    value = learn_log.data(:,param_loc);
    
    plot(value,'DisplayName',curve_names{i,1},...
        'LineWidth',1.2,'LineStyle',linestyle{i,1}, 'Color',color{i,1})
    hold on    
end

[~, param_loc] = intersect(learn_log.textdata,cellstr('val_auc'));
value = learn_log.data(:,param_loc);
plot(value,'DisplayName','Validation AUC','LineWidth',1.2,'LineStyle','-', 'Color',[0.2 0.4 1]);

legend('show')
xlabel('Iteration (Epochs)','FontName','Times','Interpreter','latex')
hold off



%%
% curve_type = {'loss';'acc';'val_loss';'val_acc'};
% curve_names = {'Train Loss';'Train Accuracy';'Validation Loss';'Validation Accuracy'};
% linestyle = {'-'; '-.'; '-'; '-.' };
% color = {[1 0.27 0.27] ; [1 0.27 0.27]; [0 0.8 0.4]; [0 0.8 0.4]};
% noCurves = size(curve_type,1);
% 
% fig = figure;
% left_color = [0 0 0];
% right_color = [0 0 0];
% set(fig,'defaultAxesColorOrder',[left_color; right_color]);
% for i = 1 : noCurves
%     yyaxis left
%     net_name = lstm_opt_model_id(1:25);
%     [param_log_name, param_log_value]  = importfilecsv([net_name, '.csv']);
%     learn_log = importdata([net_name, '.log']);
%     diagram_type_train = curve_type{i,1};
%     [~, param_loc] = intersect(learn_log.textdata,cellstr(diagram_type_train));
%     value = learn_log.data(:,param_loc);
%     
%     plot(value,'DisplayName',curve_names{i,1},...
%         'LineWidth',1.2,'LineStyle',linestyle{i,1}, 'Color',color{i,1})
%     hold on
% end
% ylabel('Loss/Accuracy');
% 
% yyaxis right
% [~, param_loc] = intersect(learn_log.textdata,cellstr('lr'));
% value = learn_log.data(:,param_loc);
% plot(value,'DisplayName','Learning Rate','LineWidth',1.2,'LineStyle','-', 'Color',[0 0 0]);
% ylabel('Learning Rate');
% ylim([10e-4 10e-3])
% legend('show')
% hold off




