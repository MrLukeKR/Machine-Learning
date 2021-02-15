function [ best_feature, best_threshold , bestGain] = chooseAttribute( features,targets )
    %Get entropic value for each node 
    %  Try n positions for each attribute to get split via maximising info
    %  gain
    
    p = sum(targets(:,1) == 1);
    n = sum(targets(:,1) == 0);
    
    I = -((p/(p+n))*log2(p/(p+n)))-((n/(p+n))*log2(n/(p+n)));
    
    best_threshold = 0;
    bestGain = 0;
    best_feature = 0;
    
    for i = 1 : size(features,2) % for each attrib
        for j = 1 : size(features,1) % for each sample
            curr_thresh = features(j,i);
            
            nL = find(features(:,i) <  curr_thresh);
            nR = find(features(:,i) >= curr_thresh);
            
            Lpos = sum(targets(nL,1) == 1);
            Lneg = sum(targets(nL,1) == 0);
            
            Rpos = sum(targets(nR,1) == 1);
            Rneg = sum(targets(nR,1) == 0);
            
            calcL = (Lpos + Lneg) / (p + n);
            calcR = (Rpos + Rneg) / (p + n);
            
            if Rpos == 0 || Rneg == 0
                impR = 0;
            else
                impR = -((Rpos/(Rpos+Rneg))*log2(Rpos/(Rpos+Rneg)))-((Rneg/(Rpos+Rneg))*log2(Rneg/(Rpos+Rneg))); 
            end
            
            if Lpos == 0 || Lneg == 0
                impL = 0;
            else
                impL = -((Lpos/(Lpos+Lneg))*log2(Lpos/(Lpos+Lneg)))-((Lneg/(Lpos+Lneg))*log2(Lneg/(Lpos+Lneg)));
            end
                    
            remainder = (calcL * impL) + (calcR * impR);
            
            newGain = I - remainder;
            
            if newGain > bestGain
                bestGain = newGain;
                best_feature = i;
                best_threshold = curr_thresh;
            end
        end      
    end
end