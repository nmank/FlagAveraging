% test_inverse_Cartan_MMG

d = 4;
l = 3;
Q = make_data_O_d(d+l,1);

repeat = 1;
% the decomposition
for j=1:repeat
    Q = make_data_O_d(d+l,1);
    [mu1, mu2, B] = inverse_Cartan_MMG(Q, d);
    norm(Q-Psi_lambda_MMG({mu1, mu2, B}, 1));
    if or(det(mu1)<0,det(mu2)<0)
        disp('haha');
    end
end

%B = B/norm(B);

% the transform
lambda = 400;
QQ = Psi_lambda_MMG({mu1, mu2, B}, lambda);
%A = cell(1,3);
%[A{1,1}, A{1,2}, A{1,3}] = inverse_Cartan_MMG(QQ, d, lambda);
A = inverse_Psi_MMG(QQ, d, lambda);
CompareMMGElements(A,{mu1, mu2, B})
