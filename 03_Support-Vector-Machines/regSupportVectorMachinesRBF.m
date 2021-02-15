clear

[points, labels, samples, dimensionality] = loadRegressionData;

KFolds = 10;
FoldSize = fix(samples / KFolds);

indices = randperm(KFolds);
predictedOutputs = zeros(FoldSize, KFolds);
expectedOutputs = zeros(FoldSize, KFolds);

bestRBF = zeros(10,3);

for rep = 1 : KFolds %Perfom K iterations for cross-validation
    
    [trainingSetX, trainingSetY, testingSetX, testingSetY] = getCrossValidationSets(indices, KFolds, FoldSize, points, labels);
    
    minErrorRBF = Inf;
    expectedOutputs(:, rep) = testingSetY;
    
    for j = 0.1:0.1:1 %Box
        for k = 0.1:0.1:1 %epsilon
            for l = 0.1:0.1:1 %kernelScale
                Mdl_RBF_r = fitrsvm(trainingSetX, trainingSetY, 'KernelFunction', 'rbf', 'BoxConstraint', j, 'KernelScale', l, 'epsilon', k);
                predictionsRBF = predict(Mdl_RBF_r,testingSetX);
                
                errorsRBF = gsubtract(testingSetY, predictionsRBF);
                SquaredErrorsRBF = errorsRBF .^ 2;
                MSE_RBF = mean(SquaredErrorsRBF);
                RMSE_RBF = sqrt(MSE_RBF);
                
                if RMSE_RBF < minErrorRBF
                    predictedOutputs(:, rep) = predictionsRBF;
                    minErrorRBF = RMSE_RBF;
                    bestRBF(rep,:) = [j,k,l];
                end
            end
        end
    end
    
    indices = circshift(indices, 1); %Cycle index order,
end

[ MSEs, RMSEs, AMSE, ARMSE ] = calculateRegressionErrors( expectedOutputs, predictedOutputs, KFolds, FoldSize);

disp("-----------------------------------------");
disp("Average Mean Squared Error: " + AMSE);
disp("Average Root Mean Squared Error: " + ARMSE);
disp("-----------------------------------------");