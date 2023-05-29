function [ estimations ] = sync_SEk_by_PD_contraction( Affinity_mat, confidence_weights, k, lambda, SO_sync_fun )
% The synchronization algorithm over SE(k), based on SVD contraction.
% This version of implementation uses synchronization in SO(k+1) via the
% eigenvectors method
% 
% Input: 
%   Affinity_mat       - upper blocks matrix of ratio measurments,(k+1)n X (k+1)n
%   confidence_weights - matrix of confidence weights for each measurement,
%                        of order nXn
%   k                  - the parameter k from SE(k)
%   lambda             - the parameter of contraction
%
% N.S, April 2016

if nargin<5
    SO_sync_fun = @Eigenvectors_Sync_SOk;
end

% the affinity matrix
s = size(Affinity_mat,1);
n = size(confidence_weights,1);
ord = k+1;
if n*(ord)~=s
    error('wrong matrices sizes');
end

% construct the affinity matrix in SO(k+1) using Psi_lambda by tranform
% each block
SO_affinity = zeros(s);
for l=1:n
    for j=(l+1):n
        if confidence_weights(l,j)>0
            ind1 = 1+(l-1)*(ord);
            ind2 = 1+(j-1)*(ord);
            mu = Affinity_mat(ind1:(ind1+k-1),ind2:(ind2+k-1));
            b  = Affinity_mat(ind1:(ind1+k-1),ind2+k);
            SO_affinity(ind1:(ind1+k),ind2:(ind2+k)) = PD_lambda(mu, b, lambda);            
%             gamma_lambda = [mu,b/lambda ; zeros(1,k),1];
%             % mapping to SO(k+1)
%             [u, ~, v] = svd(gamma_lambda);
%             q = u*v';
%             if det(q)>0
%                 SO_affinity(ind1:(ind1+k),ind2:(ind2+k)) = q;
%             else
%                 SO_affinity(ind1:(ind1+k),ind2:(ind2+k)) = u*blkdiag(eye(k),det(q))*v';
%             end
        end
    end
end
SO_affinity = SO_affinity + SO_affinity';
c = 1;%p;
SO_affinity = SO_affinity+c*eye(n*(ord));

% call the sync in SO(k+1)
[rotations_array ] = SO_sync_fun( SO_affinity, confidence_weights, ord );

% apply inverse Psi_lambda
% first, we make sure that the last element is as large as possible to
% avoid numerical troubles in the inverse PD contraction
[~,m_ind] = max(sum(abs(rotations_array(ord:ord:end,1:ord,:)),3));
if m_ind~=ord
    perm = eye(ord);
  %  idx = 1:ord;
    if m_ind==1
        idx = [2,ord,3:k,m_ind];
  %      o_ind = 1; 
    else
        idx = [k+1,2:(m_ind-1),1,(m_ind+1):k,m_ind];
  %      o_ind = 1;
    end
    perm = perm(:,idx);
    if rotations_array(ord,m_ind,1)<0
        perm(:,ord) = perm(:,ord)*(-1);
        perm(:,m_ind) = perm(:,m_ind)*(-1);
    end
    for j=1:n
    rotations_array(:,:,j) = rotations_array(:,:,j)*perm;
    end
else if rotations_array(ord,ord,1)<0
        perm = diag([ones(1,(ord-2)),-1,-1]);
        for j=1:n
            rotations_array(:,:,j) = rotations_array(:,:,j)*perm;
        end
    end
end


    
    
estimations = zeros(size(rotations_array));
for j=1:n
    estimations(:,:,j) = inverse_PD_contraction(rotations_array(:,:,j), lambda);
end

end

