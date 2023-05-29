function [ H ] = MakeAffinityMatrixOutliers(SEk_array, p, noise_func, parms)
% Construct the affinity matrix from SE data, with noise given by the
% function noise_func and its parameters, given by parms
%
% N.S. May 2016

n = size(SEk_array,3);
d = size(SEk_array,1)-1;

m = n*(n-1)/2;             % full graph upper triangular
non_outliers = floor(p*m); % number of outliers 
y = randsample(m,non_outliers); 
prob_arr = sparse(n,n);
idx = find(~tril(ones(n)));  % indices of upper side blocks
prob_arr(idx(y))=1;          % mark only the relevant, non-outliers

if nargin<4
    parms = [];
end

if nargin<3
    noise_func = @(x) eye(d+1);
end


H = zeros(n*(d+1));

for l=1:n
    for m=(l+1):n
        ind1 = 1+(l-1)*(d+1);
        ind2 = 1+(m-1)*(d+1);
        if prob_arr(l,m)
            SE_measurement = SEk_array(:,:,l)*inverse_SE_k(SEk_array(:,:,m));
            H(ind1:(ind1+d),ind2:(ind2+d))= SE_measurement;
            H(ind2:(ind2+d),ind1:(ind1+d)) = inverse_SE_k(SE_measurement);
        else   % outlier
            SE_measurement = uniform_random_SE_d(d); 
            H(ind1:(ind1+d),ind2:(ind2+d)) = SE_measurement;
            H(ind2:(ind2+d),ind1:(ind1+d))= inverse_SE_k(SE_measurement);
        end
    end
end
H = H + eye(n*(d+1));

end

