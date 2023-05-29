function [ SEk_array ] = make_data_SE_d(n,d)
% Make a series of data elements from SE(d)

SEk_array = zeros(d+1,d+1,n);
for l=1:n
    [q, ~] = qr(rand(d));
    q = q*q;
    % embed in the matrix
    SEk_array(1:d,1+d,l) = rand(d,1);
    SEk_array(1+d,1+d,l) = 1;
    SEk_array(1:d,1:d,l) = q;
 end


end