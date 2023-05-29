% script name: "LambdaVSMLE_noise"

n = 100;
d = 3;
toSave = 0;
GT_data = make_data_SE_d(n,d);

parms.d = d;
noise_func = @WrappedGaussianSE;
parms.sig1 = .35; parms.sig2 = .4;
weights = ones(n);
[Affin_mat_n1, snr1] = MakeAffinityMatrix(GT_data, weights, noise_func, parms);

%---- setup
lambdaValues = [2:20, 22:5:40];

len = numel(lambdaValues);
mse_errors2 = zeros(len,1);

ispd = 1;
%---- main loop
for j=1:len
    disp(['itration no ',num2str(j), ' out of ',num2str(len)]);
    [mse_errors2(j)] = exact_cost_function(lambdaValues(j), Affin_mat_n1, weights, GT_data, ispd);
end
nameit = ['noisyLambda_MSE_snr',num2str(ceil(snr1))];
PlotSaveLambdaVsMLE(lambdaValues, mse_errors2,  nameit, toSave);

% % second noise level
% parms.sig1 = .2; parms.sig2 = .2;
% [Affin_mat_n2, snr2] = MakeAffinityMatrix(GT_data, weights, noise_func, parms);
% 
% %---- setup
% lambdaValues = [4.8:15, 17:3:30];
% 
% len = numel(lambdaValues);
% errors2 = zeros(len,1);
% est_errors = zeros(len,1);
% mse_errors3 = zeros(len,1);
% 
% ispd = 1;
% %---- main loop
% for j=1:len
%     disp(['itration no ',num2str(j), ' out of ',num2str(len)]);
%     [~, mse_errors3(j)] = exact_cost_function(lambdaValues(j), Affin_mat_n2, weights, GT_data, ispd);
% end
% nameit = 'noisyLambda_MSE_highnoise';
% PlotSaveLambdaVsMLE(lambdaValues, mse_errors3,  nameit);
% snr1, snr2

