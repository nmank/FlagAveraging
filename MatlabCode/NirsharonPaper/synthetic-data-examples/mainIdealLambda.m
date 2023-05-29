% script name: "mainIdealLambda"
% we compare the MLE and EIG in context of choosing the ideal lambda value
% for synchronization via contraction
%
% NS, Sep 2016

tic
% make random data elements
d  = 3;
n  = 100;
GT = make_data_SE_k(n, d);
weights = ones(n);

% prepare 6 levels of noise
noiseLevels = 1:9;
len         = numel(noiseLevels);
snrLevels   = zeros(numel(noiseLevels),1);
DataMat = cell(len,1);
sig1   = .05;
sig2   = .05;
currentParm.d =  d;
noise_func    = @WrappedGaussianSE;

% lambdaValues    = [3:3:10, 20:15:200];
% LambdaValuesLen = numel(lambdaValues);
% mle_errors = zeros(len,LambdaValuesLen);
% eig_errors = zeros(len,LambdaValuesLen);
IdealMLE = zeros(len,2);
IdealEIG = zeros(len,2);
% main loop
for j=1:len
    disp(['iteration no ',num2str(j), ' out of ',num2str(len)]);
    currentParm.sig1 = sig1*noiseLevels(j);
    currentParm.sig2 = sig2*noiseLevels(j);
	[currentDataMat, snrLevels(j)] = MakeAffinityMatrix(GT, weights, noise_func, currentParm);
    DataMat{j} = currentDataMat;
    [IdealMLE(j,1) , IdealMLE(j,2)] = findIdealValue(currentDataMat, weights, GT, @sync_SO_by_maximum_likeliwood);
    [IdealEIG(j,1) , IdealEIG(j,2)] = findIdealValue(currentDataMat, weights, GT, @Eigenvectors_Sync_SOk);

    % calculating errors
%     for l=1:LambdaValuesLen
%         [~, mle_errors(j,l)] = SyncErrors(currentDataMat, weights, lambdaValues(l), GT, @sync_SO_by_maximum_likeliwood);
%         [~, eig_errors(j,l)] = SyncErrors(currentDataMat, weights, lambdaValues(l), GT, @Eigenvectors_Sync_SOk);
%     end
end
pTime = toc;

% saving results
save('DataMat','DataMat')
save('snrLevels','snrLevels')
save('IdealEIG','IdealEIG')
save('IdealMLE','IdealMLE')

% plotting
figure;
plot(snrLevels, IdealMLE(:,1), 'r','LineWidth',4);
hold on;
plot(snrLevels, IdealEIG(:,1), '--b','LineWidth',3.8);
xlabel('SNR');
ylabel('Ideal \lambda');
legend('MLE','EIG','Location','best');
set(gca,'FontSize',27);
%set(gcf,'Position',get(0,'ScreenSize'));