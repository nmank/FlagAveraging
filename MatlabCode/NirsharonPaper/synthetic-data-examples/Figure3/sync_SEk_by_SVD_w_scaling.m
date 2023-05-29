function [ estimations ] = sync_SEk_by_SVD_w_scaling(Affinity_mat, W, k, lambda)
% This function wrapps "sync_SEk_by_SVD" to allow scaling on the translational parts
%
% NS Dec 2017

n = size(W,1);
new_affinity = zeros(n);

% scaling the input
for l=1:n
    for j=1:n
        if W(j,l)~=0
            ind1 = (1+(l-1)*(k+1)):(l*(k+1));
            ind2 = 1+(j-1)*(k+1):(j*(k+1));
            current_element = Affinity_mat(ind1,ind2);
            current_element(1:k,k+1) =  current_element(1:k,k+1)/lambda;
            new_affinity(ind1,ind2) = current_element;
        end
    end
end

% run "sync_SEk_by_SVD"
[ scaled_estimation] = sync_SEk_by_SVD(new_affinity, W, k);

% rescaling back the result
estimations = zeros(k+1,k+1,n);
for j=1:n
    estimations(:,:,j) = scaled_estimation(:,:,j);
    estimations(1:k,k+1,j) = estimations(1:k,k+1,j)*lambda;
end

end

