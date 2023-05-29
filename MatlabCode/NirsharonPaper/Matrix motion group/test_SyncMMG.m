% script name: "test_SyncMMG"
%
% Run the "SyncMMG" procedure
%
% N.S, July 2017
clear;

n = 40;
d = 3; l=2;

%----- synthetic data ------
MMG_array = make_data_MMG(d, l, n);

% test 1 -- no noise
W = ones(n);

% add noise
noise_func = @(parm) { make_O_noise(d, parm.sig1) ,  make_O_noise(l, parm.sig2) , rand(d,l)*parm.sig3};
parm.sig1 = 0;
parm.sig2 = 0; %1;
parm.sig3 = 0; %1;
%H = MakeAffinityMatrixMMG(MMG_array, W, noise_func, parm);
H = MakeAffinityMatrixMMG_V1(MMG_array, W, noise_func, parm);


estimations_sepV1 = syncMMG_Separation_V1(H, W);
err_sepV1 = error_calc_MMG_V2(estimations_sepV1, MMG_array)


lambda = 50;

estimations = SyncMMG_contraction(H, W, lambda);
[err_cont, shift_cont] = error_calc_MMG_V2(estimations, MMG_array);
err_cont

estimationsV1 = SyncMMG_contractionV1(H, W, lambda);
[err_contV1, shift_contV1] = error_calc_MMG_V2(estimationsV1, MMG_array);
err_contV1

estimations_sep = syncMMG_Separation(H, W);
err_sep = error_calc_MMG_V2(estimations_sep, MMG_array)

estimations_sepV1 = syncMMG_Separation_V1(H, W);
err_sepV1 = error_calc_MMG_V2(estimations_sepV1, MMG_array)



% % ======= outliers =========
% m = n*(n-1)/2;               % full graph
% p = .8;
% non_outliers = floor(p*m);   % number of nonoutliers
% y = randsample(m,non_outliers);
% [i,j] = find(triu(ones(n),1));  % indices of upper side blocks
% I = i(y); J=j(y);
% prob_arr = sparse(I,J,ones(numel(I),1),n,n);
% 
% W = eye(n)+prob_arr+prob_arr';
% 
% % add noise
% parm.sig1 = .01;
% parm.sig2 = .0;
% parm.sig3 = 1;
% noise_func = @(parm) { make_O_noise(d, parm.sig1) ,  make_O_noise(l, parm.sig2) , rand(d,l)*parm.sig3};
% H = MakeAffinityMatrixMMG(MMG_array, W, noise_func, parm);
% %H = MakeAffinityMatrixMMG(MMG_array, W);
% 
% lambda = 100;
% 
% estimations = SyncMMG_contraction(H, W, lambda);
% err_cont = error_calc_MMG(estimations, MMG_array)
% 
% estimations_sep = syncMMG_Separation(H, W);
% err_sep = error_calc_MMG(estimations_sep, MMG_array)
% 
% % ============== OLD ================
% % W = rand(n); W(eye(n)>0)  = 1;
% % estimations = Sync_Od_spectral(A, W );
% % disp(['clean measurements, error is: ',num2str(error_calc_O_d(estimations, Od_array))])
% %
% % % test 2 -- noise on elements with small weights and uniform one (that
% % % supposed to be worst..)
% %
% % W = ones(n);
% % i = 1; j = 2;
% % W(i,j) = 0.1;
% % W(j,i) = 0.1;
% %
% % ind1 = ((i-1)*d+1):(i*d); ind2 = ((j-1)*d+1):(j1*d);
% % A(ind1,ind2) = make_data_O_d(d, 1); % outlier in the spot
% %
% % estimations1 = Sync_Od_spectral(A, W );
% % estimations2 = Sync_Od_spectral(A, ones(n));
% %
% % disp(['outlier in measurements, error with weights: ',num2str(error_calc_O_d( estimations1, Od_array))])
% % disp(['outlier in measurements, error with ones: ',num2str(error_calc_O_d( estimations2, Od_array))])
% %



