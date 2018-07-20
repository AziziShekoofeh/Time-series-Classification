%% Define an internal function to parse the values and params

function [param_value,param_loc] = parsecsv(paramname, param_log_name, param_log_value)

    [~, param_loc] = intersect(param_log_name,cellstr(paramname));
    param_value = param_log_value(param_loc, :);
    
end