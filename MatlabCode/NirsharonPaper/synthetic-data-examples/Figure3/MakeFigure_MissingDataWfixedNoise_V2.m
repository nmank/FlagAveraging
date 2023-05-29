function [errors, avg_snr, AvailData, avg_lambda] = MakeFigure_MissingDataWfixedNoise_V2(n, d)
% Ground for comparison: Full graph, ourlier, fixed noise level
% Compared methods     : Scaled spectral (SVD), 
%                        contraction (EIG for an d + MLE for d==3), 
%                        ASAP (EIG), 
%                        LS (contraction warm-up)
%
% NS, December 17

% initialization
s = (d+1)*n;               % the size
if n<150
    AvailData  = .08:.02:.2; %.08:.05:.25;        % the fraction of available data
    FixedNoise = 0.4; %.65;%
else
    AvailData  = .05:.06:.3;
    FixedNoise = .55;
end
iterations = numel(AvailData);
errors     = zeros(iterations,2);   % initiliaze errors array
avg_lambda = zeros(iterations,1);
repeatedIters = 2;                  % repeating the trial

% generate ground truth (synthetic) data
GT_data = generate_SE_array(n, d);

% use MLE for d==3
if d==3
    SO_sync_func = @sync_SO_by_maximum_likeliwood;
else
    SO_sync_func =  @Sync_SOd_spectral;
end

if repeatedIters<10
    warning(['Note. Too few iterations were set (',num2str(repeatedIters)],')');
end

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
        
        % Contraction method
        lambda_val(r) = LambdaEstimation(triu(A), W, d);
        estimations2  = SyncSEbyContraction_V2(A, W, d, lambda_val(r));
        CONT_error(r) = error_calc_SE_k(estimations2, GT_data);
        
        % Spectral method based on SVD
        estimations   = sync_SEk_by_SVD_w_scaling(triu(A), W, d, lambda_val(r));
        SPEC_error(r) = error_calc_SE_k(estimations, GT_data);
        
        % Separation method
        estimations3 = sync_SEk_by_ASAP(triu(A), W, d);
        ASAP_error(r)  = error_calc_SE_k( estimations3, GT_data);
        
        if d==3
            % Contraction method w MLE
           % lambda_val(r) = LambdaEstimation(triu(A), W, d);
            estimations_mle  =  SyncSEbyContraction_V2(A, W, d, lambda_val(r)/2, SO_sync_func);
            CONT_MLE_error(r) = error_calc_SE_k(estimations_mle, GT_data);
        end
               
        estimations4 = sync_SEk_by_MLE(triu(A), W, estimations2);
        LS_error(r) = error_calc_SE_k(estimations4, GT_data);
        
    end
    errors(q,1) = mean(SPEC_error);
    errors(q,2) = mean(CONT_error);
    errors(q,3) = mean(ASAP_error);
    errors(q,4) = mean(LS_error);
    if d==3
        errors(q,5) = mean(CONT_MLE_error);
    end
    avg_lambda(q) = mean(lambda_val);
    avg_snr = mean(snr_level);
end

%avg_snr
figure;
hold on;
plot(AvailData, errors(:,4),':','color',[0.9290    0.6940    0.1250],'LineWidth',4.2);
plot(AvailData, errors(:,3),'--k','LineWidth',3.8);
plot(AvailData, errors(:,2),'Color','blue','LineWidth',4);
plot(AvailData, errors(:,1),'-.','Color','red','LineWidth',4.3);
if d==3
    plot(AvailData, errors(:,5),':bs','LineWidth',4);
    legend('Least squares','Separation','Contraction (EIG)','Spectral','Contraction (MLE)');
else
    legend('Least squares','Separation','Contraction (EIG)','Spectral');
end
fn = sprintf('n = %d, d = %d', n, d);
title(fn);
xlabel('Fraction of avaiable data');
ylabel('MSE');
errors
hold off;
end

