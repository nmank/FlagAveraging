% script name: "test_SyncSEbyContraction"
%
% Testing "SyncSEbyContraction" 
%
% N.S, June 2016

clear; clc;
n = 100;%100;
k = 3;
s = (k+1)*n;  

lambda = 200;


%---- synthetic data in SE(k) ------
SEk_array = zeros(k+1,k+1,n);
R = 1;
for l=1:n
    % transational part
    b = R*rand(k,1); b = b/norm(b);
    % rotational part
    A = [b, rand(k,k-1)];
    [q, ~] = qr(A);
    % positive determinant
    q(:,k) = det(q)*q(:,k);
    % embed in the matrix
    SEk_array(1:k,1:k,l) = q^(mod(l,4));
    SEk_array(1:k,k+1,l) = b;
    SEk_array(k+1,k+1,l) = 1;
end

p_values = .1:.1:.3;% 0.1:0.2:.9;  % the probabilities of non-outliers
number_p_values = numel(p_values);
error_rates = zeros(number_p_values,1);

for q=1:number_p_values
    
    
    p = p_values(q);
    %p = 1; %0.6;  % the probability of non-outliers
    m = n*(n-1)/2;    % full graph
    non_outliers = floor(p*m); %number of non-outliers
    y = randsample(m,non_outliers);
    
    prob_arr = sparse(n,n);
    idx = find(~tril(ones(n)));  % indices of upper side blocks
    prob_arr(idx(y))=1;          % mark only the relevant, non-outliers
    confidence_weights = eye(n)+prob_arr+prob_arr';

   parms.d = k; parms.sig1 = 0; parms.sig2 = 0;
   noise_func = @naive_random_SE_d;
   % parms = [];
    noise_func = @(x) eye(k+1);  % Nir Sep 16
   Affin_mat = MakeAffinityMatrix(SEk_array, prob_arr, noise_func, parms);

    %---- calling the functions -----
    
    estimations = SyncSEbyContraction( triu(Affin_mat), confidence_weights, k, lambda );
    error_rates(q) = error_calc_SE_k( estimations, SEk_array );

    estimations2 = SyncSEbyContraction_V2( triu(Affin_mat), confidence_weights, k, lambda );
    error_rates2(q) = error_calc_SE_k( estimations2, SEk_array );

    %  estimations2 = SyncSEbyContraction_AVG( triu(Affin_mat), confidence_weights, k, lambda );
   % error_rates2(q) = error_calc_SE_k( estimations2, SEk_array );

end
error_rates';
error_rates2';
plot(p_values,error_rates,'r','LineWidth',3.5)
% spec = eig(Affin_mat); h1 = histogram(abs(spec),n);
hold on;
plot(p_values,error_rates2,'b','LineWidth',3.5)


