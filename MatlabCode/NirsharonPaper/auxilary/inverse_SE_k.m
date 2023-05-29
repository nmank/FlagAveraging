function [ inv_A ] = inverse_SE_k( A )
% Directly calculate the inverse by the formula
% (mu,b)^(-1) = (mu',-mu'*b)
k = size(A,1)-1;
inv_A = zeros(k+1);
inv_A(1:k,1:k) = A(1:k,1:k)';
inv_A(1:k,k+1) = (-1)*A(1:k,1:k)'*A(1:k,k+1);
inv_A(k+1,k+1) = 1;
end

