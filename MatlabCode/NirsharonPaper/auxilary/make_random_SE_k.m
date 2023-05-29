function [ rand_SE_k ] = make_random_SE_k(k)
% create random SE(k) matrix
%
% Feb 16

rand_SE_k = zeros(k+1);
[mu, ~]   = qr(randn(k)); % maybe rand ?
mu(:,k) = det(mu)*mu(:,k);
b = rand(k,1);

rand_SE_k(1:k,1:k)= mu;
rand_SE_k(1:k,1+k)= b;
rand_SE_k(1+k,1+k)= 1;

end

