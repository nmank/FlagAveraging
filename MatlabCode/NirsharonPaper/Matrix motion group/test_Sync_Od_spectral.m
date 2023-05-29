% script name: "test_Sync_Od_spectral"
%
% We test the "Sync_Od_spectral" procedure
%
% N.S, Nov 2016

n = 40;
d = 3;
s = d*n;  % the size of matrices

%---- synthetic data in O(d) ------
Od_array = make_data_O_d(d, n);

for l=1:n
    ind1 = (l*d-d+1):(l*d);   
    mu(ind1,1:d) = Od_array(:,:,l);
    mub(1:d,ind1) = Od_array(:,:,l)';
end

A = mu*mub;
%norm(A-MakeAffinityMatrixOd(Od_array,ones(n)))

% test 1 -- no noise

W = rand(n); W(eye(n)>0)  = 1;
estimations = Sync_Od_spectral(A, W );
disp(['clean measurements, error is: ',num2str(error_calc_O_d(estimations, Od_array))])

% test 2 -- noise on elements with small weights and uniform one (that
% supposed to be worst..)

W = ones(n);
i = 1; j = 2;
W(i,j) = 0.1;
W(j,i) = 0.1;

ind1 = ((i-1)*d+1):(i*d); ind2 = ((j-1)*d+1):(j*d);
A(ind1,ind2) = make_data_O_d(d, 1); % outlier in the spot

estimations1 = Sync_Od_spectral(A, W );
estimations2 = Sync_Od_spectral(A, ones(n));

disp(['outlier in measurements, error with weights: ',num2str(error_calc_O_d( estimations1, Od_array))])
disp(['outlier in measurements, error with ones: ',num2str(error_calc_O_d( estimations2, Od_array))])

% test 2 -- noisy measurements
sig_range = .1:.05:.5;
sig_plot_error = zeros(numel(sig_range),1);
for sig_ind=1:numel(sig_range)
range_d = @(x) (x*d-d+1):(x*d);
for i=1:n
    for j=(i+1):n
        A(range_d(i),range_d(j)) = Od_array(:,:,i)*(make_O_noise(d, sig_range(sig_ind)))*Od_array(:,:,j)';
    end
end
estimations_noise = Sync_Od_spectral(A, W );
sig_plot_error(sig_ind) = error_calc_O_d(estimations_noise, Od_array);
end
plot(sig_range,sig_plot_error)
%estimations_noise = Sync_Od_spectral(A, W );
%disp(['error with noise: ',num2str(error_calc_O_d(estimations_noise, Od_array))])
