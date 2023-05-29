function [ rand_SE_d ] = naive_random_SE_d(parms)
% create random SE(d) matrix with sigma variance
% (wrapped gaussian)
%
% May 16
d = parms.d;
rand_SE_d = zeros(d+1);
E = triu(randn(d),1)*parms.sig1;
E = E-E';
mu = expm(E); 
%mu(:,d) = det(mu)*mu(:,d);
b = rand(d,1)*parms.sig2;

rand_SE_d(1:d,1:d)= mu;
rand_SE_d(1:d,1+d)= b;
rand_SE_d(1+d,1+d)= 1;

end

