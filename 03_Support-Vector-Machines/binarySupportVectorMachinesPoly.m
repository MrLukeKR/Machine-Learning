clear

[points, labels, samples, dimensionality] = loadBinaryData;

KFolds = 10;
FoldSize = fix(samples / KFolds);
crossValidationSamples = FoldSize * KFolds;

indices = randperm(KFolds);
predictedOutputsPoly = zeros(FoldSize, KFolds);
expectedOutputs = zeros(FoldSize, KFolds);

bestPoly = zeros(10,2);

for rep = 1 : KFolds %Perfom K iterations for cross-validation
    
    [trainingSetX, trainingSetY, testingSetX, testingSetY] = getCrossValidationSets(indices, KFolds, FoldSize, points, labels);
    
    bestAccuracyPoly = 0;
    expectedOutputs(:, rep) = testingSetY;
    
    for j = 0.1:0.1:1 %Box
        for l = 0.1:0.1:1 %poly
            Mdl_Poly_c = fitcsvm(trainingSetX, trainingSetY, 'KernelFunction', 'polynomial', 'BoxConstraint', j, 'PolynomialOrder', l);
            predictionsPoly = predict(Mdl_Poly_c,testingSetX); %Store predicted outputs
            
            accuracyPoly = sum( predictedOutputsPoly(:, rep) == testingSetY) / size(testingSetY,1);
            
            if(accuracyPoly > bestAccuracyPoly)
                bestAccuracyPoly = accuracyPoly;
                predictedOutputsPoly(:, rep) = predictionsPoly;
                bestPoly(rep,:) = [j,l];
            end           
        end
    end
    
    indices = circshift(indices, 1); %Cycle index order,
end

[ TP, TN, FP, FN, avgAccuracy, accuracies, Precision, Recall, F1 ] = calculateBinaryAccuracy(expectedOutputs, predictedOutputsPoly, KFolds, FoldSize);

disp(avgAccuracy + "% Average Accuracy");
disp("Average Precision Rate: " + (mean(Precision)*100) + "%");
disp("Average Recall Rate: " + (mean(Recall)*100) + "%");
disp("Average F1 Rate: " + (mean(F1)*100) + "%");