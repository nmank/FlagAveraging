function [ estimations ] = sync_SEk_by_ASAP( Affinity_mat, confidence_weights, k, SO_sync_fun )

% Synchronization over SE(k), based on Cucuringu-Singer-Lipman paper:
% two stages: first, estimate solution over the rotations using the
% eigenvectors method, then use the rotational solution to derive
% translational one based on least-squares
% 
% Input: 
%   Affinity_mat       - upper blocks matrix of ratio measurments,(k+1)n X (k+1)n
%   confidence_weights - matrix of confidence weights for each measurement,
%                        of order nXn
%   k                  - the parameter k from SE(k)
%   lambda             - the parameter of contraction
%
% N.S, April 2016

if nargin<4
  % SO_sync_fun = @Eigenvectors_Sync_SOk;
   SO_sync_fun = @Sync_SOd_spectral;
end


% the affinity matrix
s = size(Affinity_mat,1);
n = size(confidence_weights,1);

if n*(k+1)~=s
    error('wrong matrices sizes');
end

% construct the affinity matrix in SO(k) 
SO_affinity = zeros(s-n);
for l=1:n
    for j=(l+1):n
        if confidence_weights(l,j)>0
            ind1 = 1+(l-1)*(k+1);
            ind1_s =  ind1-l+1  ;
            ind2 = 1+(j-1)*(k+1);
            ind2_s =  ind2-j+1  ;
            SO_affinity(ind1_s:(ind1_s+k-1),ind2_s:(ind2_s+k-1)) = Affinity_mat(ind1:(ind1+k-1),ind2:(ind2+k-1));
        end
    end
end
SO_affinity = SO_affinity + SO_affinity';
c = 1;%p;
SO_affinity = SO_affinity+c*eye(n*(k));

% call the sync in SO(k)
[rotations_array ] = SO_sync_fun( SO_affinity, confidence_weights, k );

% construct LS problem for the second part
len = nnz(triu(confidence_weights))-nnz(diag(confidence_weights));
LS_mat = zeros(len*k,s-n);
b = zeros(len*k,1);
current_line = 1;
for l=1:n
    for j=(l+1):n
        if confidence_weights(l,j)>0
            ind1 = 1+(l-1)*(k+1);
            ind2 = 1+(j-1)*(k+1);
            ind1_c = 1 + (current_line-1)*k  ;
            b(ind1_c:(ind1_c+k-1)) = Affinity_mat(ind1:(ind1+k-1),ind2+k);
                        
            ind2_l =  1 + (l-1)*k;
            LS_mat(ind1_c:(ind1_c+k-1),ind2_l:(ind2_l+k-1)) = eye(k);
            ind2_j =  1 + (j-1)*k;       
            LS_mat(ind1_c:(ind1_c+k-1),ind2_j:(ind2_j+k-1)) = (-1)*rotations_array(:,:,l)*rotations_array(:,:,j)';
            
            current_line = current_line +1;
        end
    end
end

% solve least squares
b_ls = LS_svd(LS_mat,b);  %b_ls = LS_mat\b;

% parse the solution
estimations = zeros(k+1,k+1,n);
for j=1:n
    estimations(1:k,1:k,j) = rotations_array(:,:,j);
    ind_b = (j-1)*k+1;
    estimations(1:k,k+1,j) = b_ls(ind_b:(ind_b+k-1));
    estimations(k+1,k+1,j) = 1;
end

end

