clear

[points, labels, samples, dimensionality] = loadRegressionData;

KFolds = 10;
FoldSize = fix(samples / KFolds);
crossValidationSamples = FoldSize * KFolds;

indices = randperm(KFolds);
polySVMOutputs = zeros(FoldSize, KFolds);
linearSVMOutputs = zeros(FoldSize, KFolds);
rbfSVMOutputs = zeros(FoldSize, KFolds);
annOutputs = zeros(FoldSize, KFolds);

expectedOutputs = zeros(FoldSize, KFolds);

for rep = 1 : KFolds %Perfom K iterations for cross-validation
     [trainingSetX, trainingSetY, testingSetX, testingSetY] = getCrossValidationSets(indices, KFolds, FoldSize, points, labels);
      
     expectedOutputs(:,rep) = testingSetY;
    
    %----------------------------------------------------------------------
    Mdl_Poly_r = fitrsvm(trainingSetX, trainingSetY, 'KernelFunction', 'polynomial', 'BoxConstraint', 0.1, 'PolynomialOrder', 1, 'epsilon', 0.1);
    polySVMOutputs(:,rep) = predict(Mdl_Poly_r,testingSetX);
    
    Mdl_Linear_r = fitrsvm(trainingSetX, trainingSetY, 'KernelFunction', 'linear', 'BoxConstraint', 0.1, 'epsilon', 0.9);
    linearSVMOutputs(:,rep) = predict(Mdl_Linear_r,testingSetX);
            
    Mdl_RBF_r = fitrsvm(trainingSetX, trainingSetY, 'KernelFunction', 'rbf', 'BoxConstraint', 0.1, 'KernelScale', 0.1, 'epsilon', 1);             
    rbfSVMOutputs(:,rep) = predict(Mdl_RBF_r,testingSetX);
    
    NET = newff(points',labels',10, {'tansig','purelin'}, 'trainlm', 'learngdm', 'mse'); %Create Network with given parameters
    [NET, TR] = train(NET, trainingSetX', trainingSetY'); %Train Network
    t = sim(NET, testingSetX'); %Test Network
    annOutputs(:,rep) = t';
    %----------------------------------------------------------------------
    
    indices = circshift(indices, 1); %Cycle index order
end

alpha = 0.4;

[ linMSEs, linRMSEs, linAMSE, linARMSE ] = calculateRegressionErrors( expectedOutputs, linearSVMOutputs, KFolds, FoldSize );
[ rbfMSEs, rbfRMSEs, rbfAMSE, rbfARMSE ] = calculateRegressionErrors( expectedOutputs, rbfSVMOutputs, KFolds, FoldSize );
[ polyMSEs, polyRMSEs, polyAMSE, polyARMSE ] = calculateRegressionErrors( expectedOutputs, polySVMOutputs, KFolds, FoldSize );
[ annMSEs, annRMSEs, annAMSE, annARMSE ] = calculateRegressionErrors( expectedOutputs, annOutputs, KFolds, FoldSize );

rbfAnnH = ttest2(rbfSVMOutputs, annOutputs, alpha);
linearAnnH = ttest2(linearSVMOutputs, annOutputs, alpha);
polyAnnH = ttest2(polySVMOutputs, annOutputs, alpha);