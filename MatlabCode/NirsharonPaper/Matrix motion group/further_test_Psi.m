% "further_test_Psi"
d = 4;
l = 3;
Ql = make_data_O_d(l,1);
Qd = make_data_O_d(d,1);

% M1 = [zeros(l,l), 3*pi*eye(d,l)'; -3*pi*eye(d,l), zeros(d,d)];
% M2 = [zeros(l,l), pi*eye(d,l)'; -pi*eye(d,l), zeros(d,d)];
% norm(expm(M1)-expm(M2))
% %M1*M2-M2*M1
% B1 = 3*pi*eye(d,l);
% B2 = pi*eye(d,l);
% %B1*B2'-B2*B1'
[q1, ~] = qr(rand(d));
[q2, ~] = qr(rand(l));
d1 = diag(ones(d,1));
B1 = q1*d1(:,1:l)*q2;
d2 = diag(ones(d,1)+2*pi);
B2 = q1*d2(:,1:l)*q2;
B1*B2'-B2*B1';


norm(Psi_lambda_MMG({Qd, Ql, B1},1)-Psi_lambda_MMG({Qd, Ql, B2},1))
Q = Psi_lambda_MMG({Qd, Ql, B1},1);
A = inverse_Psi_MMG(Q, d, 1);
norm(A{1,2}-Ql)
[AA1, AA2, AA3] = inverse_Cartan_MMGV1(Q, d, 1);
norm(AA1-Qd)
norm(AA2-Ql)
norm(AA3-B1)

% QQ = Psi_lambda_MMG({Qd, Ql, B2},10);
% AA = inverse_Psi_MMG(QQ, d, 10);
% norm(AA{1,3}-B2)


