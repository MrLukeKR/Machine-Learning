function tree = TrainTree( features, targets )

%op is a label for the corresponding node
%kids is a cell array that contains subtrees
%class is a label for the returning class
tree = struct('op',[],'threshold',[], 'gain',[], 'kids',[],'class', []);
       
%TRAINTREE Summary of this function goes here
%   Detailed explanation goes here
arrMean = mean(targets(:,1));

noOfSamples = size(features,1);

if ((arrMean == 0) || (arrMean == 1))
    tree.class = targets(1,1);
else
    [best_feature, best_threshold, gain] = chooseAttribute(features,targets);
    tree.op = [best_feature , gain];
    tree.threshold = best_threshold;
    tree.gain = gain;
    featuresR = [];
    featuresL = [];
    targetsR = [];
    targetsL = [];
    for i = 1 : noOfSamples
       if (features(i, best_feature) < best_threshold)
           featuresL = [featuresL;features(i,:)];
           targetsL = [targetsL;targets(i,:)];
       elseif(features(i,best_feature) >= best_threshold)
           featuresR = [featuresR;features(i,:)];
           targetsR = [targetsR;targets(i,:)];
       end
    end
    
    if(size(featuresL, 1) == 0 || size(featuresR, 1) == 0)
        tree.class = mode(targets, 1);
    else
        tree.kids = {TrainTree(featuresL, targetsL), TrainTree(featuresR, targetsR)};
    end
end
end

