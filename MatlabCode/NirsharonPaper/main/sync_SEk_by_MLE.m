function [ estimations ] = sync_SEk_by_MLE(A, W, estimation0)

% Synchronization over SE(d), based on least-squares minimization over the
% group
%
% Input:
%   A - upper blocks matrix of ratio measurments,(d+1)n X (d+1)n
%   W - matrix of confidence weights for each measurement,
%                        of order nXn
% N.S, June 2017

% basic definitions
n = size(W,1);
d = size(A,1)/n - 1;

% Create the problem structure in ManOpt
manifold  = specialeuclideanfactory(d,n);  % in R^d, n copies. Pay attention to the gradient
problem.M = manifold;

% Define the problem cost function and its Euclidean gradient.
problem.cost  = @(x) MLE_SEd_cost(x, A, W, n, d);
problem.egrad = @(x) MLE_SEd_grad(x, A, W, n, d);

% checkgradient(problem);

% initial guess
if nargin<3
    x0.t = zeros(d,n);%rand(d,1);
    for j=1:n
       % q = rand(d); [q, ~] = qr(q); q = q*q;
        x0.R(:,:,j) = eye(d);
    end
else
    for j=1:n
        x0.R(:,:,j) = estimation0(1:d,1:d,j);
        x0.t(:,j) = estimation0(1:d,1+d,j);
    end
end



% Solve.
warning('off', 'manopt:getHessian:approx');
options.tolgradnorm = 1e-10;
options.maxiter = 100;
noprint = 0;
if noprint
    options.verbosity = 0;
end
[x, xcost, info, ~] = trustregions(problem, x0,options);

% construct the solution
estimations = zeros(d+1,d+1,n);
for j=1:n
    estimations(1:d,1:d,j) = x.R(:,:,j);
    estimations(1:d,1+d,j) = x.t(:,j);
    estimations(1+d,1+d,j) = 1;
end

end

function[cost] = MLE_SEd_cost(X, A, W, n, d)
weight = 1/d;
cost = 0;
for i=1:n
    for j=(i+1):n
        if W(i,j)>0
            ind1 = 1+(i-1)*(d+1);
            ind2 = 1+(j-1)*(d+1);
            b_ij = A(ind1:(ind1+d-1),ind2+d);
            mu_ij = A(ind1:(ind1+d-1),ind2:(ind2+d-1));
            term1 = .5*(norm(mu_ij*X.R(:,:,j)-X.R(:,:,i),'fro')^2);
            % second term
            term2 = .5*norm(X.R(:,:,i)*X.R(:,:,j)'*X.t(:,j)-(X.t(:,i)-b_ij))^2;
            cost = cost + term1 + weight*term2;
        end
    end
end

end
function[grad] = MLE_SEd_grad(X, A, W, n, d)
weight = 1/d;
% initializing
grad.R = zeros(size(X.R));
grad.t = zeros(size(X.t));

for i=1:n
    for j=(i+1):n
        if W(i,j)>0
            ind1 = 1+(i-1)*(d+1);
            ind2 = 1+(j-1)*(d+1);
            b_ij = A(ind1:(ind1+d-1),ind2+d);
            mu_ij = A(ind1:(ind1+d-1),ind2:(ind2+d-1));
            %term1 = norm(mu_ij*X.R(:,:,j)-X.R(:,:,l),'fro')^2;
            grad.R(:,:,i) = grad.R(:,:,i) - (mu_ij*X.R(:,:,j)-X.R(:,:,i));
            grad.R(:,:,j) = grad.R(:,:,j) + (mu_ij'*mu_ij*X.R(:,:,j)-mu_ij'*X.R(:,:,i));
            
            % second term
            mu_i = X.R(:,:,i);  mu_j = X.R(:,:,j);
            b_i  = X.t(:,i);  b_j = X.t(:,j);
            vi = mu_j'*b_j;
            vj = mu_i'*(b_i-b_ij);
            gi = grad.R(:,:,i);   % problem with the 3D array, using auxilary array
            gj = grad.R(:,:,j);
            for p=1:d
                gi(p,:) = gi(p,:)+ vi'*(mu_i(p,:)*vi)*weight;
                gj(p,:) = gj(p,:)+ vj'*(mu_j(p,:)*vj)*weight;
            end
            grad.R(:,:,i) = gi;
            grad.R(:,:,j) = gj;
            grad.R(:,:,i) = grad.R(:,:,i) - ((b_i-b_ij)*vi')*weight;
            grad.R(:,:,j) = grad.R(:,:,j) - (b_j*vj')*weight;
            grad.t(:,j) = grad.t(:,j) + (b_j - mu_j*mu_i'*(b_i-b_ij))*weight;
            grad.t(:,i) = grad.t(:,i) - (mu_i*mu_j'*b_j - (b_i-b_ij))*weight;
        end
    end
end

end

