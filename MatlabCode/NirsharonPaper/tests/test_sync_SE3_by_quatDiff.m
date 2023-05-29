% test_sync_SE3_by_quatDiff
%
% N.S, Sep 2016

n = 10;
k = 3;

new_data = 1;%-1;

if new_data
    clear('SEk_array');
    %---- synthetic data in SE(k) ------
    translationsScale = .3; %.5;
    SEk_array = make_data_SE_k(n,k,translationsScale);
    SEk_array = makeData_SE3_withSO2Rotation(n,translationsScale);    
else
    if exist('SEk_array.mat')
        load('SEk_array.mat');
    end
end


%---- construct the similarity matrix, OUTLIERS model ----
%---- These are the "real" measurements ------------------
%---------------------------------------------------------
s = (k+1)*n;  % the size

p = 1;%0.6;  % the probability of non-outliers
m = n*(n-1)/2;    % full graph
non_outliers = floor(p*m); %number of outliers
y = randsample(m,non_outliers);

% the outliers places
prob_arr = sparse(n,n);
idx = find(~tril(ones(n)));  % indices of upper side blocks
prob_arr(idx(y))=1;          % mark only the relevant, non-outliers
confidence_weights = eye(n)+prob_arr+prob_arr';

% added noise. set to zero sig1 and sig2 for no noise
parms.d = k;
parms.sig1 = 0;%.2;%.3;
parms.sig2 = 0;%.5;%.1;
noise_func = @naive_random_SE_d;
% another option for no noise...
% parms = [];
% noise_func = @(x) eye(k+1);
Affin_mat = MakeAffinityMatrix(SEk_array, prob_arr, noise_func, parms);

% apply sync
estimations = sync_SE3_by_quatDiff( triu(Affin_mat), confidence_weights );

inv_GT = zeros(k+1,k+1,n);
for j=1:n
    inv_GT(:,:,j) = inverse_SE_k(SEk_array(:,:,j));
end
    SO_array  = inv_GT(1:k,1:k,1:n);  % debugging

disp(['SO_error_is: ', num2str(error_calc_SO_k(SO_array,estimations(1:k,1:k,:)))]);
%current_err = error_calc_SE_k( estimations, SEk_array )
current_err_W_inv = error_calc_SE_k( estimations, inv_GT )

% fliped_GT = zeros(k+1,k+1,n);
% for j=1:n
%     fliped_GT(:,:,n-j+1) = (SEk_array(:,:,j));
% end
% error_calc_SE_k( estimations, fliped_GT )
