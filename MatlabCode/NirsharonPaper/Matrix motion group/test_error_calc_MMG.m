% test_error_calc_MMG

d = 3;
l = 5;
n = 10;
A = make_data_MMG(d, l, n);
shift = make_data_MMG(d, l, 1);
B = cell(n,3);
for j=1:n
    B(j,:) = MMG_action(A(j,:),shift);
end

[e,s] = error_calc_MMG(A,B)
a_prod = MMG_action(shift,MMG_inv(s));
% the inverse and action error is
norm([norm(a_prod{1,1}-eye(d)),norm(a_prod{1,2}-eye(l)),norm(a_prod{1,3})])


