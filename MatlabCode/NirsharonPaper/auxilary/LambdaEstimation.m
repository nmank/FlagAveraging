function [ lambda_val ] = LambdaEstimation(Affin_mat, confidence_weights, d)
% Given affinity matrix of measurements, we conduct several circules of
% brute force search for a good choice of lambda
%
% NS June 2016

NumberOfRounds = 1;
NumOfInitEvals = 10;
DefualtLeap = 20;
%HardCodedVal = 0.02;

% get fundamental statistics on the transitional parts
Rel_aff = triu(confidence_weights,1);
len     = nnz(Rel_aff);
[Cind1, Cind2,~] = find(Rel_aff);
bArr    = zeros(d,len);
% getting the translations
for l=1:len
    ind1 = 1+(Cind1(l)-1)*(d+1); range1 = ind1:(ind1+d-1);
    ind2 = 1+(Cind2(l)-1)*(d+1); range2 = ind2+d;
    bArr(:,l) = Affin_mat(range1,range2);
end
norms    = sqrt(sum(bArr.*bArr));
LocalMin = mean(norms);
startP   = ceil(min(norms)*3.5);  % lambda>startP insures the method is well-defined

%[Avg, Varb] = VarOfTransPart( Affin_mat, confidence_weights, d );

% rethinking is needed here
LeapSize = max(DefualtLeap,floor((startP-LocalMin)/NumOfInitEvals));

startP = max(startP,LocalMin-NumOfInitEvals*LeapSize);
endP   = startP + NumOfInitEvals*LeapSize;

% main loop
for r=1:NumberOfRounds
    CurrentSearchArray = floor(linspace(startP,endP,NumOfInitEvals));
    error_rates        = zeros(NumOfInitEvals,1);
    for l=1:NumOfInitEvals
        error_rates(l) = EstimateSEsyncError( triu(Affin_mat), confidence_weights, d, floor(CurrentSearchArray(l)) );
    end   
    [~, ind] = min(error_rates);
    LocalMin = floor(CurrentSearchArray(ind));
    LeapSize = floor((CurrentSearchArray(2)-CurrentSearchArray(1))/4);
    startP = max(startP,LocalMin-NumOfInitEvals*LeapSize);
    endP   = LocalMin+NumOfInitEvals*LeapSize;
  %  figure; plot(CurrentSearchArray,error_rates)
end

lambda_val = LocalMin;

end

