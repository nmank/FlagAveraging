function [ estimations ] = projecting_SE(U, k, Affinity_mat, confidence_weights)
%
n = size(U,1)/(k+1);

% initial arrays
estimations = reshape(U',k+1,k+1,n);  
estimations2 = zeros(size(estimations));

% parsing and rounding
for i = 1:n
    estimations(:,:,i) = estimations(:,:,i)';
    estimations(k+1,:,i) = [zeros(1,k),1];
    estimations2(:,k+1,i) = estimations(:,k+1,i);

    % the orthogonal part
    B = estimations(1:k,1:k,i);
    [u, ~, v] = svd(B);
    estimations(1:k,1:k,i) = u*diag([diag(eye(k-1));det(u*v')])*v'; %
    estimations2(1:k,1:k,i) = u*v'*diag([diag(eye(k-2));det(u*v');1]); %m
    estimations2(:,k+1,i) = estimations(:,k+1,i);
end

if nargin>2
    err1 = estimate_SE_error_by_data(estimations, Affinity_mat, confidence_weights);
    err2 = estimate_SE_error_by_data(estimations2, Affinity_mat, confidence_weights);
    
    if err2<err1
        estimations = estimations2;
    end
end


end

