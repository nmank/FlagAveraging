function[x, fval] = findIdealValue(currentDataMat, weights, GT, soSyncfunc)
a = 1;
b = 250;
f = @(lambd) SyncErrors(currentDataMat, weights, lambd, GT, soSyncfunc);
%options = optimset('MaxFunEvals',12,'Display','iter');
options = optimset('MaxFunEvals',15,'Display','off');
[x,fval,~] = fminbnd(f, a, b, options);
end
