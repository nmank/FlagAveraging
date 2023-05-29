function [ err ] = EstimateSEsyncError(Affinity_mat, confidence_weights, k, lambda, SO_sync_fun )
% We solve the sync. problem after mapping to SO and retunr the value of
% the least squares cost function (to be minimized..)
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
    SO_sync_fun = @Eigenvectors_Sync_SOk;
end

% the affinity matrix
s = size(Affinity_mat,1);
n = size(confidence_weights,1);

if n*(k+1)~=s
    error('wrong matrices sizes');
end

% construct the affinity matrix in SO(k+1) using Psi_lambda by tranform
% each block
SO_affinity = zeros(s);
len = nnz(confidence_weights);
[Cind1, Cind2,~] = find(confidence_weights);

% loop for mapping the measurements
for l=1:len
    if Cind1(l)<Cind2(l)
        ind1 = 1+(Cind1(l)-1)*(k+1); range1 = ind1:(ind1+k);
        ind2 = 1+(Cind2(l)-1)*(k+1); range2 = ind2:(ind2+k);
        SO_affinity(range1,range2) = Psi_lambda_mat(Affinity_mat(range1,range2), lambda);
    end
end

SO_affinity = SO_affinity + SO_affinity'+eye(n*(k+1));

% call the sync in SO(k+1)
[rotations_array ] = SO_sync_fun( SO_affinity, confidence_weights, k+1 );

% shift rotations
rotations_array = shift_rotations(rotations_array);

% the computational heavy part is done by sub-sampling (bootstrap style)
bootstrap_iters = 3;
number_of_samples = min(8,nnz(triu(confidence_weights,1)));
estimated = zeros(bootstrap_iters,1);

%SE_elements = zeros(k+1,k+1,numel(relevant_element));
%ind_list    = zeros(numel(relevant_element),1);
new_place=1;
ind_list = [];
[I,J] = find(triu(confidence_weights,1));
m = numel(I);

for j=1:bootstrap_iters
   % disp(['bootstrapping no ',num2str(j)]);
    y = randsample(m,number_of_samples);
    ind1 = I(y); ind2 = J(y);
    relevant_element = unique([ind1,ind2]);  
    
    for l=1:numel(relevant_element)
        % if we didn't invert this element yet, we do it now
        if nnz((ind_list-relevant_element(l)))==numel(ind_list)
            ind_list(new_place) = relevant_element(l);
            SE_elements(:,:,new_place) = Inverse_Psi_Lambda_Rod(rotations_array(:,:,relevant_element(l)), lambda);
            new_place = new_place + 1;
        end
    end
    %error calculation
    for l=1:number_of_samples
        range1 = ((ind1(l)-1)*(k+1)+1):((ind1(l))*(k+1));
        range2 = ((ind2(l)-1)*(k+1)+1):((ind2(l))*(k+1));
        ratio_Mat = Affinity_mat(range1,range2);
        i1 = find(ind_list==I(y(l)),1);  %  find(relevant_element==I(y(l)),1);  
        M1 = SE_elements(:,:,i1);
        j1 = find(ind_list==J(y(l)),1);  %  same as above
        M2 = inverse_SE_k( SE_elements(:,:,j1) );
        estimated(j) = estimated(j) + norm(M1*M2-ratio_Mat,'fro')^2; 
    end
    estimated(j) = estimated(j)/number_of_samples;
end


err = mean(estimated);


end

