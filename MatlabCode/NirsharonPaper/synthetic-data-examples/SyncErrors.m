function [ mse ] = SyncErrors(Aff_mat, weights, lambda, GT, RotationsSync)
% Calculate the MSE and the cost function for error analysis proposes
iferr = 0;
% priliminaries
n = size(weights,1);
d = size(Aff_mat,1)/n-1;

if nargin<5
    if d==3
        RotationsSync = @sync_SO_by_maximum_likeliwood;
    else
        RotationsSync = @Eigenvectors_Sync_SOk;
    end
end

% calling the sync func
estimations = SyncSEbyContraction_V2(Aff_mat, weights, d, lambda, RotationsSync);

% calculate the errors
mse = error_calc_SE_k( estimations, GT );

% cost function
% err_cost = 0;
% if iferr
%     len = nnz(weights);
%     [Cind1, Cind2,~] = find(weights);
%     
%     % loop for mapping the measurements
%     for l=1:len
%         if Cind1(l)<Cind2(l)
%             ind1 = 1+(Cind1(l)-1)*(d+1); range1 = ind1:(ind1+d);
%             ind2 = 1+(Cind2(l)-1)*(d+1); range2 = ind2:(ind2+d);
%             err_cost = err_cost+ norm(Aff_mat(range1,range2)-estimations(:,:,Cind1(l))*inverse_SE_k(estimations(:,:,Cind2(l))),'fro')^2;
%         end
%     end
%     err_cost = err_cost/n;
% else
%     err_cost = mse;
% end
end

