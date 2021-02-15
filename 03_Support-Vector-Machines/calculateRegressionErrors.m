function [ MSEs, RMSEs, AMSE, ARMSE ] = calculateRegressionErrors( expectedOutputs, predictedOutputs, KFolds, FoldSize )
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
ARMSE = sqrt(AMSE);
end

