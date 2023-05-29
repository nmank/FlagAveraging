% "test_sync_SEk_by_SVD_w_scaling" tests the scaled version of the spectral method.

n = 40;
d = 3;
s = (d+1)*n;  % the size of matrices


%---- synthetic data in SE(k) ------
SEk_array = zeros(d+1,d+1,n);
R = 1;  % scale for transitive part data. can ignore.
% make data 
for l=1:n
    A = 3*rand(d);
    [q, ~] = qr(A);
    % positive determinant
    q(:,d) = det(q)*q(:,d);
    % transational part
    b = R*rand(d,1);
    % embed in the matrix
    SEk_array(1:d,1:d,l) = q;
    SEk_array(1:d,d+1,l) = b;
    SEk_array(d+1,d+1,l) = 1;
    
    ind1 = (l*(d+1)-d):(l*(d+1));   
    mu(ind1,1:(d+1)) = SEk_array(:,:,l);
    mub(1:(d+1),ind1) = inverse_SE_k(SEk_array(:,:,l));
end

aff_mat = mu*mub;
save('mu','mu');
% test 1 -- no noise

confidence_weights = ones(n); confidence_weights(eye(n)>0)  = 1; 
%confidence_weights = diag(sum(confidence_weights,2).^(-1))*confidence_weights;
Affin_mat = MakeAffinityMatrix(SEk_array, confidence_weights);

estimations = sync_flag(Affin_mat, confidence_weights, 3, 100);

error_calc_SE_k( estimations, SEk_array )