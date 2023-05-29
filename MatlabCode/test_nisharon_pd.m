% mu is rotation
% b is translation
k=3;

%parameter of contraction
lambda = 20;

A = rand(k);
[mu, ~] = qr(A);
% positive determinant
mu(:,k) = det(mu)*mu(:,k);

b = rand(k,1);

Q = se_to_so(mu, b ,lambda);

[new_mu, new_b] = so_to_se(Q, lambda);