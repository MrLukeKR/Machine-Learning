function tree = TrainTree( features, targets )

%op is a label for the corresponding node
%kids is a cell array that contains subtrees
%class is a label for the returning class
tree = struct('op',0,'kids',[],'class',null(0));
        
%TRAINTREE Summary of this function goes here
%   Detailed explanation goes here
arrMean = mean(targets(:,1));


if ((arrMean == 0) || (arrMean == 1))
    tree.class = targets(1,1);
    return;
else
    [best_feature, best_threshold] = chooseAttribute(features,targets);
    tree.op = best_feature;
    %{examplesi , targetsi} <- {elements of examples with best = ?i and corresponding targets}
    featuresR = [];
    featuresL = [];
    targetsR = [];
    targetsL = [];
    for i = 1:size(features, 1)
       if (features(i, best_feature) < best_threshold)
           featuresL = [featuresL;features(i,:)];
           targetsL = [targetsL;targets(i,:)];
       else
           featuresR = [featuresR;features(i,:)];
           targetsR = [targetsR;targets(i,:)];
       end
    end
    
    kidTree1 = struct('op',0,'kids',[],'class',null(0));
    kidTree2 = struct('op',0,'kids',[],'class',null(0));
    
    if(size(featuresL, 1) == 0)
        label = mode(targetsL, 1);
        kidTree1 = struct('op',best_feature,'kids',[],'class',label);
    else
        kidTree1 = TrainTree(featuresL, targetsL);
    end
    if(size(featuresR, 1) == 0)
        label = mode(targetsR);
        kidTree2 = struct('op',best_feature,'kids',[],'class',label);
    else
        kidTree2 = TrainTree(featuresR, targetsR);
    end
    
    kidArray = [kidTree1, kidTree2]
    tree.kids = kidArray;
end
    
end

