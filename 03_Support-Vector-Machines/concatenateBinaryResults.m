function concatenatedTable = concatenateBinaryResults( TP, TN, FP, FN, Accuracies, Precision, Recall, F1 )
concatenatedTable = [TP;TN;FP;FN;Accuracies;Precision;Recall;F1]';
end

