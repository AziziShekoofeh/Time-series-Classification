function [TP, FP, TN, FN, sensitivity, specificity, misClassified] = findStatResult(Predicted_Label,Label)


% Predicted_Label = (Predicted_Label == 2 | Predicted_Label == 3);
% Label = (Label == 2 | Label == 3);

TP = sum(((Predicted_Label == Label) & (Predicted_Label == 1)));
FP = sum((Predicted_Label ~= Label) & (Predicted_Label == 1));
TN = sum((Predicted_Label == Label) & (Predicted_Label == 0));
FN = sum((Predicted_Label ~= Label) & (Predicted_Label == 0));

find((Predicted_Label ~= Label) & (Predicted_Label == 1));
find((Predicted_Label ~= Label) & (Predicted_Label == 0));

sensitivity = TP/(TP+FN);
specificity = TN/(TN+FP);

misClassified = find(Predicted_Label ~= Label);

end