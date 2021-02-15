function [ trainingSetX, trainingSetY, testingSetX, testingSetY ] = getCrossValidationSets( indices, KFolds, FoldSize, points, labels )
%GETKFOLDS Summary of this function goes here
%   Detailed explanation goes here

    trainingSetX = [];
    trainingSetY = [];
    %Do partitioning of data for cross validation
    for i = 1 : KFolds
        StartInd = indices(i) * FoldSize - FoldSize + 1;
        EndInd = indices(i) * FoldSize;
        if i < KFolds
            trainingSetX = vertcat(trainingSetX, points(StartInd:EndInd,:));
            trainingSetY = vertcat(trainingSetY, labels(StartInd:EndInd,:));
        else
            testingSetX = points(StartInd:EndInd,:);
            testingSetY = labels(StartInd:EndInd,:);
        end
    end
end