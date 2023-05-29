function [errors, avg_snr, AvailData, avg_lambda] = MakeFigure_MissingDataWfixedNoise_V1(n, d)
% Ground for comparison: Full graph, ourlier, fixed noise level
% Compared methods     : Spectral (SVD), contraction, ASAP, LS
%
%
% NS, August 17

% initialization
s          = (d+1)*n;               % the size
if n<150    
    AvailData  = .08:.05:.28;        % the fraction of available data
    FixedNoise = 0.32;
else
    AvailData  = .05:.05:.3; 
    FixedNoise = 0.4;
end
iterations = numel(AvailData);
errors     = zeros(iterations,2);   % initiliaze errors array
avg_lambda = zeros(iterations,1);
repeatedIters = 2;                  % repeating the trial

% generate ground truth (synthetic) data
GT_data = generate_SE_array(n, d);

% main loop
for q=1:iterations
    parms.d    = d;
    parms.sig1 = FixedNoise;
    parms.sig2 = FixedNoise;
    noise_func = @WrappedGaussianSE;
    p = AvailData(q);
    m = n*(n-1)/2;               % full graph
    non_outliers = floor(p*m);   % number of nonoutliers
    
    % calling the functions        
    LS_error   = zeros(repeatedIters,1);
    SPEC_error = zeros(repeatedIters,1);
    CONT_error = zeros(repeatedIters,1);
    ASAP_error = zeros(repeatedIters,1);
    snr_level  = zeros(repeatedIters,1);
    lambda_val = zeros(repeatedIters,1);

    for r=1:repeatedIters
        
        % setting measurements entries (structure of available data)      
        y = randsample(m,non_outliers);
        [i,j] = find(triu(ones(n),1));  % indices of upper side blocks
        I = i(y); J=j(y);
        prob_arr = sparse(I,J,ones(numel(I),1),n,n);        
        W = eye(n)+prob_arr+prob_arr';      
        [A, snr_level(r)] = MakeAffinityMatrix(GT_data, prob_arr, noise_func, parms, 0);
        
        % Spectral method based on SVD
        estimations   = sync_SEk_by_SVD(triu(A), W, d);
        SPEC_error(r) = error_calc_SE_k(estimations, GT_data);
        
        % Contraction method
        lambda_val(r) = 60; %LambdaEstimation(triu(A), W, d);
        estimations2  =  SyncSEbyContraction_V2(A, W, d, lambda_val(r));
        CONT_error(r) = error_calc_SE_k(estimations2, GT_data);
        
        % Separation method
        estimations3 = sync_SEk_by_ASAP(triu(A), W, d);
        ASAP_error(r)  = error_calc_SE_k( estimations3, GT_data);
        
        % LS method
        estimations4 = sync_SEk_by_MLE(triu(A), W, estimations2);
        LS_error(r) = error_calc_SE_k(estimations4, GT_data);
    end
    errors(q,1) = mean(SPEC_error);
    errors(q,2) = mean(CONT_error);
    errors(q,3) = mean(ASAP_error);
    errors(q,4) = mean(LS_error);
    avg_lambda(q) = mean(lambda_val);
    avg_snr = mean(snr_level);
end

%avg_snr
figure;
hold on;
plot(AvailData, errors(:,4),':','color',[0.9290    0.6940    0.1250],'LineWidth',4.2);
plot(AvailData, errors(:,3),'--k','LineWidth',4);
plot(AvailData, errors(:,2),'Color','blue','LineWidth',4.2);
plot(AvailData, errors(:,1),'-.','Color','red','LineWidth',4.7);
legend('Least squares','Separation','Contraction','Spectral');
fn = sprintf('n = %d, d = %d', n, d);
title(fn);
xlabel('Fraction of avaiable data');
ylabel('MSE');
hold off;
end

