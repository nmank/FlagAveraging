% script name: "cost_function_illustration"

clear; clc;
n = 80;
d = 3;

% create ground truth
SEk_array = make_data_SE_d(n,d);
W = ones(n);
    
%[low, medium, higher] level of noise
noise_values = [0.1, 0.5, 0.9];               
number_n_values = numel(noise_values);

% error arrays
error_rates   = zeros(4,number_n_values);
cost_func_val = zeros(4,number_n_values);
lambda_val    = zeros(1,number_n_values);

for q=1:number_n_values
    
    % generate data
    parms.d = d; parms.sig1 = noise_values(q); parms.sig2 = noise_values(q);
    noise_func = @naive_random_SE_d;
    Affin_mat = MakeAffinityMatrix(SEk_array, W, noise_func, parms);    
    
    % contraction
    lambda_val(q)    =  LambdaEstimation(Affin_mat, W, d); %100;
    x0_cont          = SyncSEbyContraction(Affin_mat, W, d, lambda_val(q));
    error_rates(1,q) = error_calc_SE_k(x0_cont, SEk_array);
    cost_func_val(1,q) = JustCostFunc(x0_cont, Affin_mat, W, n ,d);
    [est_after_cont, ~, ~] = sync_LS_SE_d_cost_function_testing(Affin_mat, W, x0_cont, 20);
    error_rates(2,q) = error_calc_SE_k(est_after_cont, SEk_array);
    cost_func_val(2,q) = JustCostFunc(est_after_cont, Affin_mat, W, n ,d);

    % initila guess
    x0_rand          = make_data_SE_d(n,d);
    error_rates(3,q) = error_calc_SE_k(x0_rand, SEk_array);
    cost_func_val(3,q) = JustCostFunc(x0_rand, Affin_mat, W, n ,d);
    [est_after_rand, ~, ~] = sync_LS_SE_d_cost_function_testing(Affin_mat, W, x0_rand, 20);
    error_rates(4,q) = error_calc_SE_k(est_after_rand, SEk_array);
    cost_func_val(4,q) = JustCostFunc(est_after_rand, Affin_mat, W, n ,d);
 
end  
