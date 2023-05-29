% script name: "test_Sync_SOd_spectral"
%
% We test the "Sync_SOd_spectral" procedure
%
% N.S, Nov 2016

n = 40;
d = 3;
s = d*n;  % the size of matrices


%---- synthetic data in SO(k) ------
SOd_array = zeros(d,d,n);
% make data 
for l=1:n
    A = 3*rand(d);
    [q, ~] = qr(A);
    % positive determinant
    q(:,d) = det(q)*q(:,d);
    SOd_array(1:d,1:d,l) = q;
    ind1 = (l*d-d+1):(l*d);   
    mu(ind1,1:d) = SOd_array(:,:,l);
    mub(1:d,ind1) = SOd_array(:,:,l)';
end

aff_mat = mu*mub;
save('mu','mu');
% test 1 -- no noise

confidence_weights = rand(n); confidence_weights(eye(n)>0)  = 1;
estimations = Sync_SOd_spectral( aff_mat, confidence_weights );
disp(['clean measurements, error is: ',num2str(error_calc_SO_k( estimations, SOd_array))])

% test 2 -- noise on elements with small weights and uniform one (that
% supposed to be worst..)

confidence_weights = ones(n);
i1 = 1; j1 = 2;
confidence_weights(i1,j1) = 0.1;
confidence_weights(j1,i1) = 0.1;

ind1 = (i1*(d)-d+1):(i1*(d)); ind2 = (j1*(d)-d+1):(j1*(d));
[q, ~] = qr(A);
aff_mat(ind1,ind2) = q*q; % outlier in the spot

estimations1 = Sync_SOd_spectral( aff_mat, confidence_weights );
estimations2 = Eigenvectors_Sync_SOk( aff_mat, ones(n), d );

disp(['outlier in measurements, error with weights: ',num2str(error_calc_SO_k( estimations1, SOd_array))])
disp(['outlier in measurements, error with ones: ',num2str(error_calc_SO_k( estimations2, SOd_array))])




