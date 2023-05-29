function [ b_ls ] = LS_solver( confidence_weights, rotations_array, bij )
% construct LS problem for the second part of iterative separation

d = size(rotations_array,1);
n = size(rotations_array,3);
len = nnz(triu(confidence_weights))-nnz(diag(confidence_weights));

LS_mat = zeros(len*d,n*d);
b      = zeros(len*d,1);

current_line = 1;
for l=1:n
    for j=(l+1):n
        if confidence_weights(l,j)>0
            %ind1 = 1+(l-1)*(k+1);
            %ind2 = 1+(j-1)*(k+1);
            ind1_c = 1 + (current_line-1)*d  ;
            b(ind1_c:(ind1_c+d-1)) =  bij(l,j,:); %Affinity_mat(ind1:(ind1+k-1),ind2+k);
                        
            ind2_l =  1 + (l-1)*d;
            LS_mat(ind1_c:(ind1_c+d-1),ind2_l:(ind2_l+d-1)) = eye(d);
            ind2_j =  1 + (j-1)*d;       
            LS_mat(ind1_c:(ind1_c+d-1),ind2_j:(ind2_j+d-1)) = (-1)*rotations_array(:,:,l)*rotations_array(:,:,j)';
            
            current_line = current_line +1;
        end
    end
end

% solve least squares
b_ls = LS_svd(LS_mat,b);  %b_ls = LS_mat\b;
b_ls = reshape(b_ls, d, []);
end

