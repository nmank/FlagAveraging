function[mu1, mu2, B] = inverse_Cartan_MMG(Q,d,varbose)
% We reveal the orthogonal matrix x which relate to
%  the Cartan decomposition of Q, by
%      Q = exp([0_d, B^T; -B , 0_l])*blkdiag(mu2,mu1)   (*)
%
% Input:
%   Q - a matrix from O(d+l).
% Output:
%   B - see (*), from M(d,l).
%
% NS, June 2017

if nargin<3
    varbose = 0;
end

n = size(Q,1);
l = n-d;

U1 = [eye(l),zeros(l,d)]; %rand(n);  %
U2 = [zeros(l,d);eye(d)]; %rand(n);  %

m_func = @(B) [-B';eye(size(B,1))]*[B,eye(size(B,1))]+...
    [B';zeros(size(B,1))]*[B,zeros(size(B,1))]-blkdiag(zeros(size(B,2)),eye(size(B,1)));

%============= just for gradient checking!! ===========
% Create the problem structure.
manifold = euclideanfactory(d,l);
problem.M = manifold;

% Define the problem cost function and its Euclidean gradient.

problem.cost  = @(B) cost_fun(B,Q,U1,U2,m_func); 
problem.egrad = @(B) grad_fun(B,Q,U1,U2,m_func);


%=================== working examples ==========================
%problem.egrad = @(B) B*(U2*U2'); % JUST for B*U2
%problem.egrad = @(B) U1'*U1*B*U2*U2'; % JUST for U1*B*U2
%problem.egrad = @(B) dexpm(B',expm(B)); % JUST for .5*norm(expm(B),'fro')^2;
%problem.egrad = @(B) dexpm(B',U1'*(U1*expm(B)*U2)*U2'); % for .5*norm(U1*expm(B)*U2,'fro')^2;
%problem.egrad = @(B) 2*B;  % for .5*norm((m_func(B)),'fro')^2;
%=================== working examples ==========================
%checkgradient(problem);

options.tolgradnorm = 1e-14;
options.verbosity = varbose;
options.maxiter = 250;
warning('off', 'manopt:getHessian:approx')
[B, xcost, info, options] = trustregions(problem, zeros(d,l), options);

A = expm(m_func(B))*Q; %HERE, NIR 1214
% % get solution on principle branch
% if norm(B,'fro')>pi
%   Big_B  = logm(A*Q'); % maybe like this?  P = logm(Q*blkdiag(R',1));
%   B = Big_B((l+1):n,1:l);
% end

% concluding
mu2 = U1*A*[eye(l);zeros(d,l)];
mu1 = [zeros(d,l),eye(d)]*A*U2;

% ============== supporting functions ================
    function[val] = cost_fun(B,Q,U1,U2,m_func)
        val = .5*norm(U1*(expm(m_func(B))*Q)*U2,'fro')^2 + ...
            .5*norm(U2'*(expm(m_func(B))*Q)*U1','fro')^2;
    end

    function[gr_expm] = grad_fun(B,Q,U1,U2,m_func)
        [d1,l1] = size(B);
        n1 = d1 + l1;
        R = U1*(expm(m_func(B))*Q)*U2;
        W = dexpm(m_func(-B),U1'*R*(Q*U2)');
        gr_expm = W((l1+1):n1,1:l1)-W(1:l1,(l1+1):n1)';
        R2 = U2'*(expm(m_func(B))*Q)*U1';
        W2 = dexpm(m_func(-B),U2*R2*(Q*U1')');
        gr_expm = gr_expm + W2((l1+1):n1,1:l1)-W2(1:l1,(l1+1):n1)';             
    end
end
