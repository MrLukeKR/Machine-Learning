%Binary Classification
clear;
clc;

load 'facialPoints.mat';
load 'labels.mat';

topology = [3, 5];
activationFuncs =   {'purelin','purelin', 'purelin'}; % Hidden Layers, Output Layer
trainingFunc = 'trainlm';
learningFunc = 'learngdm';
errorFunc = 'mse';

samples = size(points,3);
dimensionality = size(points,1) * size(points,2);

points = reshape(points,[dimensionality,samples]);
labels = labels';

shuffleInputs = randperm(samples); %Shuffle inputs to reduce sampling bias

for i = 1 : samples
   shuffledPoints(:,i) = points(:, shuffleInputs(i));
   shuffledLabels(:,i) = labels(:, shuffleInputs(i));
end

points = shuffledPoints;
labels = shuffledLabels;

KFolds = 10;
FoldSize = fix(samples / KFolds);
crossValidationSamples = FoldSize * KFolds;

indices = randperm(10);
predictedOutputs = zeros(1, FoldSize, KFolds);
expectedOutputs = zeros(1, FoldSize, KFolds); 

for rep = 1 : KFolds %Perfom K iterations for cross-validation

    trainingSetX = [];
    trainingSetY = [];
    
    %Do partitioning of data for cross validation
    for i = 1 : KFolds
        StartInd = indices(i) * FoldSize - FoldSize + 1;
        EndInd = indices(i) * FoldSize;
        if i < KFolds
            trainingSetX = horzcat(trainingSetX, points(:,StartInd:EndInd));
            trainingSetY = horzcat(trainingSetY, labels(:,StartInd:EndInd));
        else
            testingSetX = points(:,StartInd:EndInd);
            testingSetY = labels(:,StartInd:EndInd);
        end
    end
    
    %----------------------------------------------------------------
    
    %PARAMS: (FeatureVec, LabelVec, [NeuronsLayer1, ...,
    %NeuronsLayerN], {ActivationFuncLayer1, ..., ActivationFuncLayerN},
    %trainFunc, learnFunc, errorCalcFunc);
    
    NET = newff(points,labels,topology, activationFuncs, trainingFunc, learningFunc, errorFunc); %Create Network with given parameters
    
    %NET.trainParam.lr=0.1;
    
    NET.trainParam.epochs = 10000;
    
    %NET.performParam.regularization = 0.1;
    
    %----------------------------------------------------------------
    
    [NET, TR] = train(NET, trainingSetX, trainingSetY); %Train Network
    
    t = sim(NET, testingSetX); %Test Network
    
    predictedOutputs(:,:, rep) = t; %Store predicted outputs
    expectedOutputs(:,:, rep) = testingSetY;
    
    indices = circshift(indices, 1); %Cycle index order,
end

predictedOutputs = reshape(predictedOutputs,[1, crossValidationSamples]);
expectedOutputs = reshape(expectedOutputs,  [1, crossValidationSamples]);

compare = [expectedOutputs;predictedOutputs];

disp("-----------------------------------------");

errors = gsubtract(expectedOutputs, predictedOutputs);
c = confusion(expectedOutputs, predictedOutputs);

TP = zeros(1,i);
TN = zeros(1,i);
FP = zeros(1,i);
FN = zeros(1,i);
Precision = zeros(1,i);
Recall = zeros(1,i);
F1 = zeros(1,i);

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

accuracy = (1 - c) * 100;

disp(accuracy + "% Average Accuracy");

disp("-----------------------------------------");
disp("Use the 'compare' matrix to view Prediction vs. Expected");

%Train final net on ALL data
NET = newff(points,labels,topology, activationFuncs, trainingFunc, learningFunc, errorFunc); %Create Network with given parameters %Create Network with given parameters
[NET, TR] = train(NET, points, labels); %Train Network

save binaryClassificationOutput;
