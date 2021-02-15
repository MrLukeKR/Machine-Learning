function [ points, labels, samples, dimensionality ] = loadRegressionData
%LOADREGRESSIONDATA Summary of this function goes here
%   Detailed explanation goes here
load('facialPointsR.mat')
load('labels.mat')
load('headpose.mat')
labels = pose(:,6);

samples = size(points,3);
dimensionality = size(points,1) * size(points,2);

points = reshape(points,[dimensionality,samples])';
points = points(1:samples,:);
pose = pose(1:samples,:);

shuffledInputs = randperm(samples);

for i = 1 : samples
    shuffledPoints(i,:) = points(shuffledInputs(i),:);
    shuffledLabels(i,:) = labels(shuffledInputs(i),:);
end

points = shuffledPoints;
labels = shuffledLabels;
end

