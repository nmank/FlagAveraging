function [ err, shift ] = error_calc_SE_k( estimations, true_data )
% we measure the error by (finding the group element g which minimizes)
% sum_i \| estimations_i g - true_data_i \|
%
% N.S Feb 16

% rotation shift
k = size(true_data,1)-1;
if size(estimations,1)~=k+1
    error('wrong matrix sizes');
end

n = size(true_data,3);
if size(estimations,3)~=n
    error('wrong matrix sizes');
end

A = zeros(k);
for j=1:n 
    A  = A + estimations(1:k,1:k,j)'*true_data(1:k,1:k,j);
end
[u,~,v] = svd(A);
mu_shift = u*diag([diag(eye(k-1));det(u*v')])*v';

% trasalation shift
b = zeros(k,1);
for j=1:n 
    b  = b + estimations(1:k,1:k,j)'*(true_data(1:k,1+k,j)-estimations(1:k,1+k,j));
end
b = b/n;

% conclude the shifting
shift = blkdiag(mu_shift,1);
shift(1:k,k+1) = b; 
for j=1:n 
    estimations(:,:,j)  = estimations(:,:,j)*shift;
end

%error calculations
err = 0;
for j=1:n
    err = err + norm(estimations(:,:,j)-true_data(:,:,j),'fro')^2; 
end
err = err/n;
end

