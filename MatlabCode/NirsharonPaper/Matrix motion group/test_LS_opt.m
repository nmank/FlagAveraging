% script name: "test_LS_opt"

d = 4;
l = 3;
n = 10;

% ground truth
for j=1:n
    X(:,:,j) = rand(d,l);
end

% data construction
Aij = cell(n,n);
Bij = cell(n,n);
Cij = cell(n,n);
for i=1:n
    for j=1:n
        Aij{i,j} = rand(d);
        Bij{i,j} = rand(l);
        Cij{i,j} = (-1)*(Aij{i,j}*X(:,:,j) +  X(:,:,i)*Bij{i,j});
    end
end

% Aij{1,1} = rand(d);
% Bij{1,1} = rand(l);
% Cij{1,1} = rand(d,l);
% Aij{2,2} = rand(d);
% Bij{2,2} = rand(l);
% Cij{2,2} = rand(d,l);
%for j=1:n X0((j*d-d+1):(j*d),:) = X(:,:,j); end
A = LS_opt(Aij, Bij, Cij);   %, X0);
norm(A(:,:,1)-X(:,:,1))