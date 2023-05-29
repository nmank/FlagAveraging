function [ err ] = estimate_error_by_data_SE(estimations, H, weights)
% Heuristicaly measuring the quality of a given solution by 
% calculating rhe cost function
%
% Feb 16
k = size(estimations,1);
n = size(estimations,3);

if nargin<3
    weights = ones(n);
end

if size(H,1)~=n*k
    error('wrong matrix size');
end

err = 0;
for l=1:n
    for j=(l+1):n
        ind1 = 1+(l-1)*k;
        ind2 = 1+(j-1)*k;
        if weights(l,j)>0
            diff = norm(estimations(:,:,l)*inverse_SE_k(estimations(:,:,j)) - H(ind1:(ind1+k-1),ind2:(ind2+k-1)),'fro');
            err = err + weights(l,j)*diff;
        end
    end
end

err = err/nnz(triu(weights));

end

