%Multiclass Classification
clear;
clc;

load emotions_data.mat;

topology = [10];
activationFuncs = {'tansig', 'softmax'}; % Hidden Layers, Output Layer
trainingFunc = 'trainlm';
learningFunc = 'learngdm';
errorFunc = 'mse';
classes = 6;

uninitpoints = x';
uninitlabels = zeros(size(y,1),classes);

samples = size(x,1);
dimensionality = size(x,2);

for i = 1 : samples %Converting real-values to column-wise class memberships
    uninitlabels(i,y(i)) = 1;
end

uninitlabels = uninitlabels';

shuffleInputs = randperm(samples); %Shuffle inputs to reduce sampling bias

for i = 1 : samples
   labels(:,i) = uninitlabels(:,shuffleInputs(i)); 
   points(:,i) = uninitpoints(:,shuffleInputs(i));
end

KFolds = 10;
FoldSize = fix(samples / KFolds);
crossValidationSamples = FoldSize * KFolds;

indices = randperm(KFolds);

predictedOutputs = zeros(classes, FoldSize, KFolds);
expectedOutputs  = zeros(classes, FoldSize, KFolds);

for rep = 1 : KFolds %Perfom K iterations for cross-validation
    trainingSetX = [];
    trainingSetY = [];
    
    %Do partitioning of data for cross validation
    for i = 1 : KFolds
        StartInd = indices(i) * FoldSize - FoldSize + 1;
        EndInd = indices(i) * FoldSize;
        if i < KFolds
            trainingSetX = horzcat(trainingSetX, points(:, StartInd:EndInd));
            trainingSetY = horzcat(trainingSetY, labels(:, StartInd:EndInd));
        else
            testingSetX = points(:, StartInd:EndInd);
            testingSetY = labels(:, StartInd:EndInd);
        end
    end
    
    %----------------------------------------------------------------
    
    %PARAMS: (FeatureVec, LabelVec, [NeuronsLayer1, ...,
    %NeuronsLayerN], {ActivationFuncLayer1, ..., ActivationFuncLayerN},
    %trainFunc, learnFunc, errorCalcFunc);
    NET = newff(points,labels,topology, activationFuncs, trainingFunc, learningFunc, errorFunc); %Create Network with given parameters
    %NET.performParam.regularization = 0.5;
    %NET.trainParam.epochs = 10000;
    
%----------------------------------------------------------------
    
    [NET, TR] = train(NET, trainingSetX, trainingSetY); %Train Network
    
    t = sim(NET, testingSetX); %Test Network
    
    predictedOutputs(:, :, rep) = t; %Store predicted outputs
    expectedOutputs(:, :, rep) = testingSetY; %Store expected outputs
   
    indices = circshift(indices, 1); %Cycle index order,
end

fixedPredictedOutputs = zeros(1,crossValidationSamples);
fixedExpectedOutputs = zeros(1,crossValidationSamples);
predictedOutputs = reshape(predictedOutputs,[classes,crossValidationSamples]);
expectedOutputs  = reshape(expectedOutputs, [classes,crossValidationSamples]);

for i = 1 : crossValidationSamples
    [val, ind] = max(predictedOutputs(:, i));
    fixedPredictedOutputs(1,i) = ind;
    
    [val, ind] = max(expectedOutputs(:, i));
    fixedExpectedOutputs(1,i) = ind;
end

compareFixed = [fixedExpectedOutputs;fixedPredictedOutputs];
compare = [expectedOutputs;predictedOutputs];

disp("-----------------------------------------");
disp("Plotting Confusion Matrix...");
 
errors = gsubtract(expectedOutputs, predictedOutputs);
c = confusion(expectedOutputs, predictedOutputs);

plotconfusion(expectedOutputs, predictedOutputs, "Average Multiclass Classification");

accuracies = zeros(1,KFolds);

for i = 1 : KFolds
   sInd = FoldSize * i - FoldSize + 1;
   eInd = FoldSize * i;
   accuracies(1,i) = (sum(compareFixed(1,sInd:eInd) == compareFixed(2,sInd:eInd)) / FoldSize) * 100;
   figure
   plotconfusion(expectedOutputs(:,sInd:eInd), predictedOutputs(:,sInd:eInd), "Multiclass Classification: Fold " + i);
end

accuracy = ((1-c) * 100);
disp(accuracy + "% Average Accuracy");
disp("-----------------------------------------");

%Train final net on ALL data
NET = newff(points,labels,topology, activationFuncs, trainingFunc, learningFunc, errorFunc); %Create Network with given parameters %Create Network with given parameters
[NET, TR] = train(NET, points, labels); %Train Network


save multiclassClassificationOutput;
