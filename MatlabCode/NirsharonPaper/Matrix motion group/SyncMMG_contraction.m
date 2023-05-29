function [ estimations ] = SyncMMG_contraction(A, W, lambda)
% Solving synchronization over MMG(d,l), based on group contraction and
% on Sync_Od_spectral O(d+l).
%
%
% Input:
%   A       - upper blocks matrix of ratio measurments,(k+1)n X (k+1)n
%   W - matrix of confidence weights for each measurement,
%                        of order nXn
%   lambda             - the parameter of contraction
%
% N.S, July 2017

% initializing sizes
n = size(W,1);
d = size(A{1,1}{1,1},1);
l = size(A{1,1}{1,2},1);

% construct affinity matrix in O(d+l) using Psi_lambda by tranform each
% element

s = n*(d+l);
O_affinity = zeros(s);
% len = nnz(W);
% [Cind1, Cind2,~] = find(W);
% 
% % mapping the avaiable measurements
% for j=1:len
%     if Cind1(j)<Cind2(j)
%         ind1 = 1+(Cind1(j)-1)*(d+l); range1 = ind1:(ind1+d+l-1);
%         ind2 = 1+(Cind2(j)-1)*(d+l); range2 = ind2:(ind2+d+l-1);
%         O_affinity(range1,range2) = Psi_lambda_MMG(A{Cind1(j),Cind2(j)}, lambda);
%     end
% end
range_dl = @(ind) (1+(ind-1)*(d+l)):(ind*(d+l));
for j=1:n
    for m=(j+1):n
        if W(j,m)>0            
            O_affinity(range_dl(j),range_dl(m)) = Psi_lambda_MMG(A{j,m}, lambda);
        else
            O_affinity(range_dl(j),range_dl(m)) = zeros(d+l);
        end
    end
end
O_affinity = O_affinity + O_affinity' + eye(s);

% call the sync in SO(k+1)
[rotations_array ] = Sync_Od_spectral(O_affinity, W);
% estimate_error_by_data(rotations_array, O_affinity, W)

% ===========================================
% option A shift it all according to first
rotations_arrayA = zeros(size(rotations_array));
for j=1:n
    rotations_arrayA(:,:,j) = rotations_array(:,:,j)*rotations_array(:,:,1)';
end

% preparing for conclusions
estimationsA = cell(n,3);
for j=1:n
    estimationsA(j,:) = inverse_Psi_MMG(rotations_arrayA(:,:,j), d, lambda);
    if norm(Psi_lambda_MMG(estimationsA(j,:),lambda)-rotations_arrayA(:,:,j))>1e-12
        disp('haha');
    end
end
errA = estimate_MMG_from_data(estimationsA, A, W );

% option B -- as comes from O(d)
estimationsB = cell(n,3);
for j=1:n
    estimationsB(j,:) = inverse_Psi_MMG(rotations_array(:,:,j), d, lambda);
end
errB = estimate_MMG_from_data(estimationsB, A, W );


% conclude
if errA<errB
    estimations = estimationsA;
else
    estimations = estimationsB;
end

end

