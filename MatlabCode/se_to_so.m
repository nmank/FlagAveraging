function [ Q ] = se_to_so(mu, b, lambda)

k=3;

% PD Contraction
gamma_lambda = [mu,b/lambda ; zeros(1,k),1];
% mapping to SO(k+1)
[u, ~, v] = svd(gamma_lambda);
Q = u*v';
% Q = v*u'; %this was the bug. incorrect polar decomposition
