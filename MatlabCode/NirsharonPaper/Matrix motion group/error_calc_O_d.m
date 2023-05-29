function [avg_sqr_err, shift_rotation] = error_calc_O_d(A, B)
% calculate the rotation to minimize the error between 
% series A and B of $n$ group elements
% 
% OUTPUT: average squared error
%
%  NS, June 17

n = size(A,3);

diff = zeros(size(A));
for j=1:n 
    diff(:,:,j)  = A(:,:,j)'*B(:,:,j);
end

P = sum(diff,3);
[u, ~, v] = svd(P);
shift_rotation = u*v';

err = 0;
for j=1:n
    err = err + norm(A(:,:,j)*shift_rotation-B(:,:,j),'fro')^2; 
end
avg_sqr_err = err/n;

end

