function [ Q ] = make_O_noise( d, sig )
%

rand_sign  = 1;%sign(rand-.5);
A = randn(d)*sig;
A = A-A';
Q = expm(A);
Q(:,1) = Q(:,1)*rand_sign;

end

