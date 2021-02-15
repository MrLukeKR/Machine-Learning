clear

[points, labels, samples, dimensionality] = loadBinaryData;

KFolds = 10;
FoldSize = fix(samples / KFolds);
crossValidationSamples = FoldSize * KFolds;

indices = randperm(KFolds);
predictedOutputsRBF = zeros(FoldSize, KFolds);
expectedOutputs = zeros(FoldSize, KFolds);

bestRBF = zeros(10,2);
bestFoldAccuracy = zeros(10,1);

for rep = 1 : KFolds %Perfom K iterations for cross-validation
    
    [trainingSetX, trainingSetY, testingSetX, testingSetY] = getCrossValidationSets(indices, KFolds, FoldSize, points, labels);
    
    bestAccuracyRBF = 0;
    expectedOutputs(:, rep) = testingSetY;
            
    for j = 0.1:0.1:1 %Box
        for l = 1:100 %kernelScale
            Mdl_RBF_c = fitcsvm(trainingSetX, trainingSetY, 'KernelFunction', 'rbf', 'BoxConstraint', 0.4, 'KernelScale', 32);
            predictionsRBF = predict(Mdl_RBF_c,testingSetX); %Store predicted outputs
           
            accuracyRBF = sum(predictedOutputsRBF(:,rep) == testingSetY) / size(testingSetY,1);
            
            if(accuracyRBF > bestAccuracyRBF)
                bestAccuracyRBF = accuracyRBF;
                predictedOutputsRBF(:, rep) = predictionsRBF;
                bestRBF(rep,:) = [j,l];
            end
            
            if(accuracyRBF > bestFoldAccuracy(rep, 1))
                bestFoldAccuracy(rep, 1) = accuracyRBF;
            end
        end
    end
    
    indices = circshift(indices, 1); %Cycle index order
end

[ TP, TN, FP, FN, avgAccuracy, accuracies, Precision, Recall, F1 ] = calculateBinaryAccuracy(expectedOutputs, predictedOutputsRBF, KFolds, FoldSize);

disp(avgAccuracy + "% Average Accuracy");
disp("Average Precision Rate: " + (mean(Precision)*100) + "%");
disp("Average Recall Rate: " + (mean(Recall)*100) + "%");
disp("Average F1 Rate: " + (mean(F1)*100) + "%");