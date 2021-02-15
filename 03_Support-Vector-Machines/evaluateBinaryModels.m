clear

[points, labels, samples, dimensionality] = loadBinaryData;

KFolds = 10;
FoldSize = fix(samples / KFolds);
crossValidationSamples = FoldSize * KFolds;

indices = randperm(KFolds);
polySVMOutputs = zeros(FoldSize, KFolds);
linearSVMOutputs = zeros(FoldSize, KFolds);
rbfSVMOutputs = zeros(FoldSize, KFolds);
annOutputs = zeros(FoldSize, KFolds);
dtOutputs = zeros(FoldSize, KFolds);

expectedOutputs = zeros(FoldSize, KFolds);

for rep = 1 : KFolds %Perfom K iterations for cross-validation
    [trainingSetX, trainingSetY, testingSetX, testingSetY] = getCrossValidationSets(indices, KFolds, FoldSize, points, labels);
    
    expectedOutputs(:,rep) = testingSetY;
    
   %--------------------------------------------------------------------------
    Mdl_Poly_c = fitcsvm(trainingSetX, trainingSetY, 'KernelFunction', 'polynomial', 'BoxConstraint', 0.1, 'PolynomialOrder', 0.1);
    polySVMOutputs(:,rep) = predict(Mdl_Poly_c,testingSetX);
    
    Mdl_Linear_c = fitcsvm(trainingSetX, trainingSetY, 'KernelFunction', 'linear', 'BoxConstraint', 0.1);
    linearSVMOutputs(:,rep) = predict(Mdl_Linear_c,testingSetX);
            
    Mdl_RBF_c = fitcsvm(trainingSetX, trainingSetY, 'KernelFunction', 'rbf', 'BoxConstraint', 0.4, 'KernelScale', 32);             
    rbfSVMOutputs(:,rep) = predict(Mdl_RBF_c,testingSetX);
    
    NET = newff(points',labels',[3,5], {'purelin','purelin', 'purelin'}, 'trainlm', 'learngdm', 'mse'); %Create Network with given parameters
    [NET, TR] = train(NET, trainingSetX', trainingSetY'); %Train Network
    t = sim(NET, testingSetX'); %Test Network
    annOutputs(:,rep) = t';
    
    Tree = TrainTree(trainingSetX,trainingSetY);
    for j = 1 : FoldSize
        dtOutputs(j, rep) = SimulateTree(Tree, testingSetX(j,:));
    end
   %--------------------------------------------------------------------------
   
    indices = circshift(indices, 1); %Cycle index order
end

alpha = 0.05;

[ annTP, annTN, annFP, annFN, annAvgAccuracy, annAccuracies, annPrecision, annRecall, annF1 ] = calculateBinaryAccuracy( expectedOutputs, annOutputs, KFolds, FoldSize );
[ rbfTP, rbfTN, rbfFP, rbfFN, rbfAvgAccuracy, rbfAccuracies, rbfPrecision, rbfRecall, rbfF1 ] = calculateBinaryAccuracy( expectedOutputs, rbfSVMOutputs, KFolds, FoldSize );
[ linTP, linTN, linFP, linFN, linAvgAccuracy, linAccuracies, linPrecision, linRecall, linF1 ] = calculateBinaryAccuracy( expectedOutputs, linearSVMOutputs, KFolds, FoldSize );
[ polyTP, polyTN, polyFP, polyFN, polyAvgAccuracy, polyAccuracies, polyPrecision, polyRecall, polyF1 ] = calculateBinaryAccuracy( expectedOutputs, polySVMOutputs, KFolds, FoldSize );
[ dtTP, dtTN, dtFP, dtFN, dtAvgAccuracy, dtAccuracies, dtPrecision, dtRecall, dtF1 ] = calculateBinaryAccuracy( expectedOutputs, polySVMOutputs, KFolds, FoldSize );

rbfAnnH = ttest2(rbfSVMOutputs, annOutputs, alpha);
rbfDtH= ttest2(rbfSVMOutputs,dtOutputs, alpha);

linearAnnH = ttest2(linearSVMOutputs,annOutputs, alpha);
linearDtH = ttest2(linearSVMOutputs,dtOutputs, alpha);

polyAnnH = ttest2(polySVMOutputs,annOutputs, alpha);
polyDtH = ttest2(polySVMOutputs,dtOutputs, alpha);