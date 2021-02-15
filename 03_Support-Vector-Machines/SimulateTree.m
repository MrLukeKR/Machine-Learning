function output = SimulateTree(Tree, testFeatures)
    while(isempty(Tree.class))
        feature = Tree.op(1);
        threshold = Tree.threshold;

         if(testFeatures(1,feature) < threshold)
            Tree = Tree.kids{1};
        else
            Tree = Tree.kids{2};
        end
    end
    
    output = Tree.class;
end