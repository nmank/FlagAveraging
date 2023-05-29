function [ Od_array ] = make_data_O_d(d, n)
% Make a series of $n$ data elements from O(d)
%
% SEE : "How to generate random matrices from the classical compact groups" 
%       by Francesco Mezzadri
%
% NS, June 17

Od_array = zeros(d,d,n);
for l=1:n
    z = randn(d)/sqrt(2.0);
    [q,r] = qr(z);
    dd = diag(r);
    ph = dd./abs(dd);
    q = q*diag(ph)*q;

    Od_array(:,:,l) = q;
 end


end