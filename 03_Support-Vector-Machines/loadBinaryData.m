function [ points, labels, samples, dimensionality ] = loadBinaryData
load('facialPointsB.mat')
load('labels.mat')

samples = size(points,3);
dimensionality = size(points,1) * size(points,2);

points = reshape(points,[dimensionality,samples])';

shuffleInputs = randperm(samples); %Shuffle inputs to reduce sampling bias

for i = 1 : samples
   shuffledPoints(i,:) = points(shuffleInputs(i),:);
   shuffledLabels(i,:) = labels(shuffleInputs(i),:);
end

points = shuffledPoints;
labels = shuffledLabels;


end

