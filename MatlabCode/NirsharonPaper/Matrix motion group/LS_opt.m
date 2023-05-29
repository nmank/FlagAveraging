function[B_array] = LS_opt(Aij, Bij, Cij, X0)
% performing the LS by gradient optimization,
%  min sum_ij\| Aij*Xj + Xi*Bij + Cij \|_F^2
% where
% Xi are n matrices of order dXl
% Aij cell array of nXn, consists of dXd matrix each
% Bij cell array of nXn, consists of lXl matrix each
% Cij cell array of nXn, consists of dXl matrix each
%
% NS, July 2017

n = size(Cij,1);
[d,l] = size(Cij{1,1});

if nargin<4
    X0 = rand(n*d,l);
end

%============= just for gradient checking!! ===========
% Create the problem structure.
manifold = euclideanfactory(n*d,l);
problem.M = manifold;

% Define the problem cost function and its Euclidean gradient.
problem.cost  = @(x) cost_fun(x, Aij, Bij, Cij, d, n);
problem.egrad = @(x) grad_fun(x, Aij, Bij, Cij, d, n);

%checkgradient(problem);

%=================== working examples ==========================
%problem.egrad = @(B) A'*(AX-B); % JUST for A*X-B
%problem.egrad = @(B) B*(U2*U2'); % JUST for B*U2
%problem.egrad = @(B) U1'*U1*B*U2*U2'; % JUST for U1*B*U2
%problem.egrad = @(B) dexpm(B',expm(B)); % JUST for .5*norm(expm(B),'fro')^2;
%problem.egrad = @(B) dexpm(B',U1'*(U1*expm(B)*U2)*U2'); % for .5*norm(U1*expm(B)*U2,'fro')^2;
%problem.egrad = @(B) 2*B;  % for .5*norm((m_func(B)),'fro')^2;
%=================== working examples ==========================

options.tolgradnorm = 1e-13;
options.maxiter = 140;
options.verbosity = 0;
warning('off', 'manopt:getHessian:approx')
[X, xcost, info, options] = trustregions(problem, X0, options);

% parse solution
B_array = zeros(d,l,n);
for j=1:n
    B_array(:,:,j) = X((1+(j-1)*d):(j*d),:);
end

% ============== supporting functions ================
    function[val] = cost_fun(x, Aij, Bij, Cij, d, n)
        val = 0;
        d_range = @(d_ind) (1+(d_ind-1)*d):(d_ind*d);
        for i=1:n
            for p=1:n
                if ~isempty(Cij{i,p})
                    val = val + .5*norm(Aij{i,p}*x(d_range(p),:)+x(d_range(i),:)*Bij{i,p}+Cij{i,p},'fro')^2;
                end
            end
        end
    end

    function[grad] = grad_fun(x, Aij, Bij, Cij, d, n)
        d_range = @(d_ind) (1+(d_ind-1)*d):(d_ind*d);
        grad = zeros(size(x));
        for i=1:n
            for p=1:n
                if ~isempty(Cij{i,p})
                    grad(d_range(p),:) = grad(d_range(p),:) + ...
                        Aij{i,p}'*(Aij{i,p}*x(d_range(p),:)+x(d_range(i),:)*Bij{i,p}+Cij{i,p});
                    grad(d_range(i),:) = grad(d_range(i),:) + ...
                        (Aij{i,p}*x(d_range(p),:)+x(d_range(i),:)*Bij{i,p}+Cij{i,p})*Bij{i,p}';
                end
            end
        end
    end
end
