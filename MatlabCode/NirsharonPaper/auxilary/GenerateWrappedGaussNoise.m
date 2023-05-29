function [ A ] = GenerateWrappedGaussNoise(parms)
% Generate noise SE matrix from Wrapped Gaussian Distribution 

d     = parms.dimn;
mu    = parms.mean;
sigma = parms.devi;

% when sigma is covariance (non-diagonal) 
%sigma = [1 0.5; 0.5 2];
%R = chol(sigma);
triu
E = repmat(mu,10,1) + randn(10,2)*R
E = mu + randn(n,

end

