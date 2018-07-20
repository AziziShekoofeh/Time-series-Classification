%   plot the learning curves for 
%   @ Code composed by Shekoofeh Azizi on 18/10/2017 (UBC-RCL)
%%
clc
clear all %#ok<CLALL>
close all

%%

model_type = 'gru';
diagram_type_train = 'loss';
diagram_type_val = 'val_loss';
compare_type = 'initial_lr';

[subFiles, opt_param_log_name, opt_param_log_value] = setting(model_type);
noFiles = size(subFiles,1);

[opt_lr, ~ ] = parsecsv('initial_lr', opt_param_log_name, opt_param_log_value);
[opt_bs, ~ ] = parsecsv('batch_size', opt_param_log_name, opt_param_log_value);
[opt_optimizer, ~ ] = parsecsv('optimizer', opt_param_log_name, opt_param_log_value);
[opt_reg, ~ ] = parsecsv('l2_regulizer', opt_param_log_name, opt_param_log_value);
[opt_do, ~ ] = parsecsv('dropout', opt_param_log_name, opt_param_log_value);

bs = '128';
do = '0';
reg = '0.0001';

fig = figure;
% Create axes
ax = axes('Parent',fig);
set(ax,'FontName','Times','FontSize',14,'GridLineStyle',':',...
    'LineStyleOrderIndex',3,'XGrid','on','YGrid','on');
hold on
for i = 1 : noFiles
    filename = subFiles{i,1};
    net_name = filename(1:25);
    [param_log_name, param_log_value]  = importfilecsv([net_name, '.csv']);
    learn_log = importdata(filename);
    
    [file_lr, ~ ] = parsecsv('initial_lr', param_log_name, param_log_value);
    [file_bs, ~ ] = parsecsv('batch_size', param_log_name, param_log_value);
    [file_optimizer, ~ ] = parsecsv('optimizer', param_log_name, param_log_value);
    [file_reg, ~ ] = parsecsv('l2_regulizer', param_log_name, param_log_value);
    [file_do, ~ ] = parsecsv('dropout', param_log_name, param_log_value);
    
    if(strcmp(bs, file_bs) && strcmp(reg, file_reg) && strcmp(do, file_do))
        
        switch(file_lr{1,1})
            case '0.01'
                linestyle = ':';
            case '0.0001'
                linestyle = '-';
            otherwise
                fprintf('Invalid!\n' );
        end
        
        switch(file_optimizer{1,1})
            case 'sgd'
                [~, param_loc] = intersect(learn_log.textdata,cellstr(diagram_type_train));
                value = learn_log.data(:,param_loc);
                plot(value,'DisplayName',strcat('sgd, lr = ', file_lr{1,1}),...
                    'LineWidth',1.2,'LineStyle',linestyle, 'Color',[0.2 0.4 1])
                hold on
                %         [~, param_loc] = intersect(learn_log.textdata,cellstr(diagram_type_val));
                %         value = learn_log.data(:,param_loc);
                %         plot(value,'DisplayName',strcat('validation loss: Lr = ', file_lr{1,1}))
                %         hold on
            case 'rmsprop'
                [~, param_loc] = intersect(learn_log.textdata,cellstr(diagram_type_train));
                value = learn_log.data(:,param_loc);
                plot(value,'DisplayName',strcat('rmsprop, lr = ', file_lr{1,1}), ...
                    'LineWidth',1.2,'LineStyle',linestyle, 'Color',[1 0.27 0.27])
                hold on
                %         [~, param_loc] = intersect(learn_log.textdata,cellstr(diagram_type_val));
                %         value = learn_log.data(:,param_loc);
                %         plot(value,'DisplayName',strcat('validation loss: Lr = ', file_lr{1,1}))
                %         hold on
            case 'adam'
                [~, param_loc] = intersect(learn_log.textdata,cellstr(diagram_type_train));
                value = learn_log.data(:,param_loc);
                plot(value,'DisplayName',strcat('adam, lr = ', file_lr{1,1}),...
                    'LineWidth',1.2,'LineStyle',linestyle,'Color', [0 0.8 0.4])
                hold on
                %         [~, param_loc] = intersect(learn_log.textdata,cellstr(diagram_type_val));
                %         value = learn_log.data(:,param_loc);
                %         plot(value,'DisplayName',strcat('validation loss: Lr = ', file_lr{1,1}))
                %         hold on
            otherwise
                fprintf('Invalid!\n' );
        end
    end
end
ylabel('Loss','FontName','Times','Interpreter','latex','FontSize',14);
xlabel('Iteration (Epochs)','FontName','Times','Interpreter','latex','FontSize',14)
ylim([0 0.7])
legend('show')
box('on')
hold off




