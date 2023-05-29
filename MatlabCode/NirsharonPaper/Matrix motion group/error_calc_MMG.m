function [avg_sqr_err, shift_element] = error_calc_MMG(A, B)
% calculate the rotation to minimize the error between 
% series A and B (cell arrays) nX3 of Matrix Motion Group elements
% 
% OUTPUT: average squared error
%
%  NS, June 17

n = size(A,1);

diff = cell(size(A));
sum1 = 0;
sum2 = 0;
sum3 = 0;
for j=1:n 
    diff(j,:)  = MMG_action(MMG_inv(A(j,:)),B(j,:));
    sum1 = sum1 + diff{j,1};
    sum2 = sum2 + diff{j,2};
    sum3 = sum3 + A{j,1}'*(B{j,3}-A{j,3})*A{j,2};
end

[u1, ~, v1] = svd(sum1);
shift_element{1,1} = u1*v1';
[u2, ~, v2] = svd(sum2);
shift_element{1,2} = u2*v2';
shift_element{1,3} = sum3/n;

% conclude the shifting
shifted_A = cell(n,3);
for j=1:n 
    shifted_A(j,:)  = MMG_action(A(j,:),shift_element);
end

%error calculations
err = 0;
for j=1:n
    err = err + norm(shifted_A{j,1}-B{j,1},'fro')^2 + ...
        norm(shifted_A{j,2}-B{j,2},'fro')^2+norm(shifted_A{j,3}-B{j,3},'fro')^2; 
end
avg_sqr_err = err/n;

end

