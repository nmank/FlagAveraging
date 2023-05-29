function [ A ] = inverse_PD_contraction(Q, lambda)
% We calculate the inverse for the PD-based contraction
%
% NS, April 2016
k = size(Q,1)-1;

theta_l = .5*Q(k+1,k+1);
b = Q(1:k,k+1)*(lambda/theta_l);
if norm(b)<eps
    mu = Q(1:k,1:k);
else
    %U = null(b');
    [~, ~, v1] = svd(b');
    U = v1(:,2:end);
    
    
    P = U*U';
    b_n = b/norm(b);
    B = (2*theta_l*b_n*b_n'+P);
    mu = inv(B)*Q(1:k,1:k);
end

% summary
A = zeros(k+1);
A(k+1,k+1) = 1;
A(1:k,k+1) = b;
A(1:k,1:k) = mu;
if norm(PD_tranform(A,lambda)-Q)>1e-13
    warning('bad inverse PD tranform');
end
end

