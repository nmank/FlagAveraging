function [ estimations ] = syncMMG_Separation(A, W)
% Solving synchronization over MMG(d,l), based on separation: Od sync and
% least squares
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

% construct the affinity matrix in O(d) 
Od = zeros(n*d);
Ol = zeros(n*l);
range_d = @(ind) (1+(ind-1)*d):(ind*d);
range_l = @(ind) (1+(ind-1)*l):(ind*l);
for j=1:n
    for m=(j+1):n
        if W(j,m)>0            
            Od(range_d(j),range_d(m)) = A{j,m}{1,1};
            Ol(range_l(j),range_l(m)) = A{j,m}{1,2};           
        end
    end
end

Od = Od + Od' + eye(n*d);
Ol = Ol + Ol' + eye(n*l);

% call the sync in SO(k)
[mu1_array] = Sync_Od_spectral(Od, W);
[mu2_array] = Sync_Od_spectral(Ol, W);

% construct LS problem for the second part
Aij = cell(n,n);
Bij = cell(n,n);
Cij = cell(n,n);
for i=1:n
    for j=1:n
        if W(i,j)>0
           Aij{i,j} = mu1_array(:,:,i)*mu1_array(:,:,j)';
           Bij{i,j} = -mu2_array(:,:,i)*mu2_array(:,:,j)';
           Cij{i,j} = A{i,j}{1,3}*mu2_array(:,:,i)*mu2_array(:,:,j)';
        end
    end
end

% solve least squares
B_array = LS_opt(Aij, Bij, Cij);

% parse the solution
estimations = cell(n,3);
for j=1:n
    estimations{j,1} = mu1_array(:,:,j);
    estimations{j,2} = mu2_array(:,:,j);
    estimations{j,3} = B_array(:,:,j);
end

end

