function[ rotations_array ] = Sync_SOd_spectral( Affinity_mat, confidence_weights, x )
% The eigenvectors synchronization algorithm in SO(k) 
% 
% Input: 
%   Affinity_mat       - blocks matrix of ratio measurments, order dn X dn.
%   confidence_weights - matrix of confidence weights for each measurement,
%                        order n X n.
%   d                  - the order of matrices, that is matrices from SO(d)
%
% N.S, Feb 2016

% the affinity matrix
s = size(Affinity_mat,1);
n = size(confidence_weights,1);
d = s/n;

% the matrix
H = zeros(s);
D = zeros(s);
for l=1:n
    for j=(l):n
        ind1 = (1+(l-1)*d):(l*d);
        ind2 = (1+(j-1)*d):(j*d);
        H(ind1,ind2) = confidence_weights(l,j)*Affinity_mat(ind1,ind2);
        H(ind2,ind1) = confidence_weights(l,j)*Affinity_mat(ind1,ind2)';
    end
end
D = sum(confidence_weights,2);  % degree of nodes = sum of weights 
D = kron(diag(D),eye(d));

% constract normalization
D_minus_half = diag( diag(D).^(-.5) );
Sym_H = D_minus_half*H*D_minus_half;

%extract eigenvectors
try
    [vecs, ~] = eigs(Sym_H,d);
catch
    [vecs, ~] = eigs(Sym_H,d); %%% Something to avoid troubles
end
vecs = D_minus_half*vecs;

% rounding -- first version
rotations_array1 = zeros(d,d,n);
if det(vecs(1:d,:))<0   % NIR May 2
    vecs(:,1) = (-1)*vecs(:,1);
end
for j=1:n
    ind1 = 1+(j-1)*d;        
    B = vecs(ind1:(ind1+d-1),:);
    % B = vecs(ind1:(ind1+k-1),k:(-1):1); %, vecs(ind1,:)*sign(vecs(ind1,k))];  % NIR check
    [u , ~, v] = svd(B);
    %rotations_array(:,:,j) = u*v'*diag([diag(eye(k-1));det(u*v')]);
    rotations_array1(:,:,j) = u*diag([diag(eye(d-2));det(u*v');1])*v'; % this is better approx
end

% rounding -- onur version
rotations_array2 = zeros(d,d,n);
R_Mat = vecs(:,d:(-1):1); %unknown why. maybe a bug
%R_Mat = vecs;
if det(R_Mat(1:d,1:d))<0
    R_Mat(:,1) = (-1)*R_Mat(:,1);
end
R_Matcut = reshape(R_Mat',d,d,n);   
for i = 1:n
    B = R_Matcut(:,:,i)';
    B(:,d) = B(:,d)*sign(B(d,d)); %unknown why (keeping the last entry positive?) CHECK THIS 29 April
    [u, ~, v] = svd(B);
    rotations_array2(:,:,i) = u*v'*diag([diag(eye(d-2));det(u*v');1]);
end

err1 = estimate_error_by_data(rotations_array1,H,confidence_weights);
err2 = estimate_error_by_data(rotations_array2,H,confidence_weights);
%err3 = estimate_error_by_data(rotations_array3,H,confidence_weights);

eps1 = 100*eps; 
if err1<(err2-eps1)
    rotations_array = rotations_array1;
else
    rotations_array = rotations_array2;
end

% % avoid numerical problems due to last element
% A = min(abs(rotations_array),[],3);
% thd = 0.15;
% if A(end,end)<thd
%     [~,indm] = max(A(end,:));
%     if indm~=k
%         per_arr = 1:k;
%         if indm>1
%             per_arr(k)=indm;
%             per_arr(indm-1)=k;
%             per_arr(indm)=indm-1;
%         else
%             per_arr(k)=indm;
%             per_arr(indm)=indm+1;
%             per_arr(indm+1)=k;
%         end
%         rotations_array = rotations_array(:,per_arr,:);
%     end
% end
        
        


end

