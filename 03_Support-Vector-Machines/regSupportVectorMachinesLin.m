clear

[points, labels, samples, dimensionality] = loadRegressionData;

samples = 150;

KFolds = 10;
FoldSize = fix(samples / KFolds);

indices = randperm(KFolds);
predictedOutputs = zeros(FoldSize, KFolds);
expectedOutputs = zeros(FoldSize, KFolds);

bestLin = zeros(10,2);

for rep = 1 : KFolds %Perfom K iterations for cross-validation
    
    [trainingSetX, trainingSetY, testingSetX, testingSetY] = getCrossValidationSets(indices, KFolds, FoldSize, points, labels);
    
    minErrorLin = Inf;
    expectedOutputs(:, rep) = testingSetY;
    
    for j = 0.1:0.1:1 %Box
        for k = 0.1:0.1:1 %epsilon
            Mdl_Linear_r = fitrsvm(trainingSetX, trainingSetY, 'KernelFunction', 'linear', 'BoxConstraint', k, 'epsilon', k);
            predictionsLinear = predict(Mdl_Linear_r,testingSetX);
            
            errorsLin = gsubtract(testingSetY, predictionsLinear);
            SquaredErrorsLin = errorsLin .^ 2;
            MSE_Lin = mean(SquaredErrorsLin);
            RMSE_Lin = sqrt(MSE_Lin);
            
            if RMSE_Lin < minErrorLin
                predictedOutputs(:, rep) = predictionsLinear;
                minErrorLin = RMSE_Lin;
                bestLin(rep,:) = [j,k];
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



