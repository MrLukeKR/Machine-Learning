function [ TP, TN, FP, FN, avgAccuracy, accuracies, Precision, Recall, F1 ] = calculateBinaryAccuracy( expectedOutputs, predictedOutputs, KFolds, FoldSize )
TP = zeros(1,KFolds);
TN = zeros(1,KFolds);
FP = zeros(1,KFolds);
FN = zeros(1,KFolds);
Precision = zeros(1,KFolds);
Recall = zeros(1,KFolds);
F1 = zeros(1,KFolds);

accuracies = zeros(1,KFolds);

for i = 1 : KFolds
   sInd = FoldSize * i - FoldSize + 1;
   eInd = FoldSize * i;
   [acc,cm] = confusion(expectedOutputs(sInd:eInd), predictedOutputs(sInd:eInd));
   
   accuracies(1,i) = (1 - acc) * 100;
   
    TP(1,i) = cm(1,1);
    TN(1,i) = cm(2,2);
    FN(1,i) = cm(1,2);
    FP(1,i) = cm(2,1);
    Precision(1,i) = (TP(1,i) / (TP(1,i) + FP(1,i)));
    Recall(1,i) = (TP(1,i) / (TP(1,i) + FN(1,i)));
    F1(1,i) = 2 * ((Precision(1,i) * Recall(1,i)) / (Precision(1,i) + Recall(1,i)));
end

avgAccuracy = mean(accuracies);
end

