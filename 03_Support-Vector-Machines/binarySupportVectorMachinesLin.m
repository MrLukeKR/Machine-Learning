clear

[points, labels, samples, dimensionality] = loadBinaryData;

KFolds = 10;
FoldSize = fix(samples / KFolds);
crossValidationSamples = FoldSize * KFolds;

indices = randperm(KFolds);
predictedOutputsLinear = zeros(FoldSize, KFolds);
expectedOutputs = zeros(FoldSize, KFolds);
bestLin = zeros(10,1);

for rep = 1 : KFolds %Perfom K iterations for cross-validation
    
    [trainingSetX, trainingSetY, testingSetX, testingSetY] = getCrossValidationSets(indices, KFolds, FoldSize, points, labels);
    
    bestAccuracyLinear = 0;
    expectedOutputs(:, rep) = testingSetY;
    
    for j = 0.1:0.1:1 %Box
        Mdl_Linear_c = fitcsvm(trainingSetX, trainingSetY, 'KernelFunction', 'linear', 'BoxConstraint', j);
        predictionsLinear = predict(Mdl_Linear_c,testingSetX); %Store predicted outputs
        
        accuracyLinear = sum( predictedOutputsLinear(:, rep) == testingSetY) / size(testingSetY,1);
        
        if(accuracyLinear > bestAccuracyLinear)
            bestAccuracyLinear = accuracyLinear;
            predictedOutputsLinear(:, rep) = predictionsLinear;
            bestLin(rep,:) = j;
        end
    end
    
    indices = circshift(indices, 1); %Cycle index order
end

[ TP, TN, FP, FN, avgAccuracy, accuracies, Precision, Recall, F1 ] = calculateBinaryAccuracy(expectedOutputs, predictedOutputsLinear, KFolds, FoldSize);

disp(avgAccuracy + "% Average Accuracy");
disp("Average Precision Rate: " + (mean(Precision)*100) + "%");
disp("Average Recall Rate: " + (mean(Recall)*100) + "%");
disp("Average F1 Rate: " + (mean(F1)*100) + "%");