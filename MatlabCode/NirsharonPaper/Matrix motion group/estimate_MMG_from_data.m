function [ err ] = estimate_MMG_from_data(estimations, H, W )
% Heuristicaly measuring the quality of a given solution by 
% calculating the cost function
%
% July 16
n = size(estimations,1);
err = 0;

if nargin<3
    W = ones(n);
end

for l=1:n
    for j=(l+1):n
         if W(l,j)>0
            diff = CompareMMGElements(MMG_action(estimations(l,:),MMG_inv(estimations(j,:)) ) , H{l,j});
            err = err + W(l,j)*diff;
        end
    end
end

err = 2*err/(n*(n-1)); % averaging the error
end