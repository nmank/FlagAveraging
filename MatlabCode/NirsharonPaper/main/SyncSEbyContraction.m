function [ estimations ] = SyncSEbyContraction(Affinity_mat, confidence_weights, d, lambda, SO_sync_fun )
% The synchronization algorithm over SE(d), based on group contraction and
% on SO_sync_fun synchronization in SO(d+1).
%
% Input:
%   Affinity_mat       - upper blocks matrix of ratio measurments,(k+1)n X (k+1)n
%   confidence_weights - matrix of confidence weights for each measurement,
%                        of order nXn
%   k                  - the parameter k from SE(k)
%   lambda             - the parameter of contraction
%   SO_sync_fun        - the function that process the SO sync (default is eigenvector method)
%
% N.S, June 2016

if nargin<5
    % SO_sync_fun = @Eigenvectors_Sync_SOk; %% OLD
    SO_sync_fun = @Sync_SOd_spectral;
end

% if d<=3
%     SO_sync_fun = @sync_SO_by_maximum_likeliwood;
% end

% initializing sizes
s = size(Affinity_mat,1);
n = size(confidence_weights,1);

if n*(d+1)~=s
    error('wrong matrices sizes');
end

% construct affinity matrix in SO(k+1) using Psi_lambda by tranform each block
SO_affinity = zeros(s);
len = nnz(confidence_weights);
[Cind1, Cind2,~] = find(confidence_weights);

% mapping the avaiable measurements
for l=1:len
    if Cind1(l)<Cind2(l)
        ind1 = 1+(Cind1(l)-1)*(d+1); range1 = ind1:(ind1+d);
        ind2 = 1+(Cind2(l)-1)*(d+1); range2 = ind2:(ind2+d);
        SO_affinity(range1,range2) = Psi_lambda_mat(Affinity_mat(range1,range2), lambda);
    end
end
SO_affinity = SO_affinity + SO_affinity'+eye(s);

% call the sync in SO(k+1)
[rotations_array ] = SO_sync_fun( SO_affinity, confidence_weights, d+1 );

%rotations_array = shift_rotations( rotations_array, rotations_array(:,:,1)' );


% applying inverse Psi_lambda.
estimations = zeros(size(rotations_array));



% PARALLEL COMPUTING when: Matlab 2015a and above AND more than 350 elements
[~, ver] = version;
% disp('done rotations sync, apply psi inverse');
if (str2double(ver(end-4:end))>=2015)&&(n>350)
    parpool
    parfor j=1:n
        estimations(:,:,j) = Inverse_Psi_Lambda_Rod(rotations_array(:,:,j), lambda);
        %estimations(:,:,j) = inverse_Psi_lambda(rotations_array(:,:,j), lambda);
    end
    poolobj = gcp('nocreate');
    delete(poolobj);
else
    for j=1:n
        estimations(:,:,j) = Inverse_Psi_Lambda_Rod(rotations_array(:,:,j), lambda);
        %estimations(:,:,j) = inverse_Psi_lambda(rotations_array(:,:,j), lambda);
    end
end
end


