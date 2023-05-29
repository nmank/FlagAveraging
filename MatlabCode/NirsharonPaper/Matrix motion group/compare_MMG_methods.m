% script name: "compare_MMG_methods"
clear;

d = 3;
l = 4;
n = 10;

MMG_array = make_data_MMG(d, l, n);

% ======= outliers =========
p = 1;%.8;                      % percentage of nonoutliers
m = n*(n-1)/2;               % full graph
non_outliers = floor(p*m);   % number of nonoutliers
y = randsample(m,non_outliers);
[i,j] = find(triu(ones(n),1));  % indices of upper side blocks
I = i(y); J=j(y);
prob_arr = sparse(I,J,ones(numel(I),1),n,n);
W = eye(n)+prob_arr+prob_arr';

% added noise
parm.sig1 = 00;
parm.sig2 = .0;
parm.sig3 = .0;
noise_func = @(parm) { make_O_noise(d, parm.sig1) ,  make_O_noise(l, parm.sig2) , rand(d,l)*parm.sig3};
H = MakeAffinityMatrixMMG(MMG_array, W, noise_func, parm);

lambda = 100;

estimations = SyncMMG_contraction(H, W, lambda);
err_cont = error_calc_MMG(estimations, MMG_array)

estimations_sep = syncMMG_Separation(H, W);
err_sep = error_calc_MMG(estimations_sep, MMG_array)
