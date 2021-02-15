%Regression
clear;
clc;

load facialPoints.mat;
load headpose.mat;
labels = pose(:,6);

topology = [25];
activationFuncs = {'tansig', 'purelin'}; % Hidden Layers, Output Layer
trainingFunc = 'trainlm';
learningFunc = 'learngdm';
errorFunc = 'mse';
classes = 6;

dimensionality = size(points,1) * size(points,2);
samples = size(points,3);

points = reshape(points,[dimensionality,samples]);
labels = labels';

shuffledInputs = randperm(samples);

for i = 1 : samples
    shuffledPoints(:,i) = points(:, shuffledInputs(i));
    shuffledLabels(:,i) = labels(:, shuffledInputs(i));
end

points = shuffledPoints;
labels = shuffledLabels;

KFolds = 10;
FoldSize = fix(samples / KFolds);
crossValidationSamples = KFolds * FoldSize;

indices = randperm(KFolds);
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
    
    %NOTE: Output layer must be purelin because this is a regression based 
    %output, not probabilistic
    NET = newff(points,labels,topology, activationFuncs, trainingFunc, learningFunc, errorFunc); %Create Network with given parameters
    %NET.trainParam.lr = 0.0001;
    %NET.performParam.regularization = 0.1;
    %NET.trainParam.epochs = 10000;
    
    %----------------------------------------------------------------
    
   [NET, TR] = train(NET, trainingSetX, trainingSetY); %Train Network
    
    t = sim(NET, testingSetX); %Test Network
    
    predictedOutputs(:,:, rep) = t; %Store predicted outputs
    expectedOutputs(:,:, rep) = testingSetY;
    
    indices = circshift(indices, 1); %Cycle index order,
end


predictedOutputs = reshape(predictedOutputs,[1, crossValidationSamples]);
expectedOutputs  = reshape(expectedOutputs, [1, crossValidationSamples]);

compare = [expectedOutputs; predictedOutputs];

errors = gsubtract(expectedOutputs, predictedOutputs);
SquaredErrors = errors .^ 2;

MSEs = zeros(size(FoldSize));
RMSEs = zeros(size(FoldSize));

for i = 1 : KFolds
   sInd = FoldSize * i - FoldSize + 1;
   eInd = FoldSize * i;
   MSEs(i) = mean(SquaredErrors(sInd:eInd)); 
   RMSEs(i) = sqrt(MSEs(i));
end

AMSE = mean(MSEs);
ARMSE = mean(RMSEs);

disp("-----------------------------------------");
disp("Average Mean Squared Error: " + AMSE);
disp("Average Root Mean Squared Error: " + ARMSE);
disp("-----------------------------------------");

%Train final net on ALL data
NET = newff(points,labels,topology, activationFuncs, trainingFunc, learningFunc, errorFunc); %Create Network with given parameters %Create Network with given parameters
[NET, TR] = train(NET, points, labels); %Train Network

save regressionOutput;
