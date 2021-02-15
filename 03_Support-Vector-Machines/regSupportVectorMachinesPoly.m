clear

[points, labels, samples, dimensionality] = loadRegressionData;

KFolds = 150;
FoldSize = fix(samples / KFolds);

indices = randperm(KFolds);
predictedOutputs = zeros(FoldSize, KFolds);
expectedOutputs = zeros(FoldSize, KFolds);
bestPoly = zeros(10,3);

for rep = 1 : KFolds %Perfom K iterations for cross-validation
    
    [trainingSetX, trainingSetY, testingSetX, testingSetY] = getCrossValidationSets(indices, KFolds, FoldSize, points, labels);
    
    minErrorPoly = Inf;
    expectedOutputs(:, rep) = testingSetY;
    
    for j = 0.1:0.1:1 %Box
        for k = 0.1:0.1:1 %epsilon
            for l = 0.1:0.1:1 %poly
                Mdl_Poly_r = fitrsvm(trainingSetX, trainingSetY, 'KernelFunction', 'polynomial', 'BoxConstraint', j, 'PolynomialOrder', l, 'epsilon', k);
                predictionsPoly = predict(Mdl_Poly_r,testingSetX);
                
                errorsPoly = gsubtract(testingSetY, predictionsPoly);
                SquaredErrorsPoly = errorsPoly .^ 2;
                MSE_Poly = mean(SquaredErrorsPoly);
                RMSE_Poly = sqrt(MSE_Poly);
                
                if RMSE_Poly < minErrorPoly
                    predictedOutputs(:, rep) = predictionsPoly;
                    minErrorPoly = RMSE_Poly;
                    bestPoly(rep,:) = [j,k,l];
                end    
            end
        end
    end
   
    indices = circshift(indices, 1); %Cycle index order,
end

[ MSEs, RMSEs, AMSE, ARMSE ] = calculateRegressionErrors(expectedOutputs, predictedOutputs, KFolds, FoldSize);

disp("-----------------------------------------");
disp("Average Mean Squared Error: " + AMSE);
disp("Average Root Mean Squared Error: " + ARMSE);
disp("-----------------------------------------");