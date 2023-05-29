function [ ] = MakeFigure_Outliers(n, d)
% Ground for comparison: Full graph, ourlier, no noise
% Compared methods     : Spectral (SVD), contraction, ASAP
%
%
% NS, June 16

% initialization
s          = (d+1)*n;                 % the size
pValues    = .4:.1:.9;% 0.3:0.14:1;             % the probabilities of non-outliers
iterations = numel(pValues);
errors     = zeros(iterations,2);   % initiliaze errors array
repeatedIters = 1;                  % Ideally 5-10

% The favorite SO procedure
if d==2
    SO_sync_func = @sync_SO_by_LUD_V2;  % LUD
else if d==3
        SO_sync_func = @sync_SO_by_maximum_likeliwood;  % MLE
    else
        %SO_sync_func = @Eigenvectors_Sync_SOk; % EIG - OLD
        SO_sync_func = @Sync_SOd_spectral; % EIG
    end
end

% generate ground truth (synthetic) data
GT_data = generate_data_SE_k(n, d, 2);

% main loop
for q=1:iterations
    % setting measurements entries (non-outliers/outliers)
    p = pValues(q);
    m = n*(n-1)/2;               % full graph
    non_outliers = floor(p*m);   % number of nonoutliers
    y = randsample(m,non_outliers);
    [i,j] = find(triu(ones(n),1));  % indices of upper side blocks
    I = i(y); J=j(y);
    prob_arr = sparse(I,J,ones(numel(I),1),n,n);
    
    confidence_weights = ones(n); %eye(n)+prob_arr+prob_arr';
    
    parms.d = d; parms.sig1 = 0; parms.sig2 = 0; % no noise
    noise_func = @naive_random_SE_d;             % OR @(x) eye(k+1);
    Affin_mat = MakeAffinityMatrix(GT_data, prob_arr, noise_func, parms, 1);
    
    
    % calling the functions
    
    SPEC_error = zeros(repeatedIters,1);
    CONT_error = zeros(repeatedIters,1);
    ASAP_error = zeros(repeatedIters,1);
    disp(['iteration no.',num2str(q)]);
    
    for r=1:repeatedIters
        % Spectral method based on SVD
        estimations1   = sync_SEk_by_SVD( triu(Affin_mat), confidence_weights, d );
        SPEC_error(r) = error_calc_SE_k( estimations1, GT_data );
        
        % Contraction method
        lambda = 200;                                        
        estimations2  =  SyncSEbyContraction_V2(Affin_mat, confidence_weights, d, lambda, SO_sync_func);
        CONT_error(r) = error_calc_SE_k(estimations2, GT_data );       

        % ASAP L1
        if n<120
            estimations3 = sync_SEk_by_ASAP_L1(triu(Affin_mat), confidence_weights, d, SO_sync_func);
        else
            estimations3 = sync_SEk_by_ASAP(triu(Affin_mat), confidence_weights, d, SO_sync_func);
        end
        ASAP_error(r)  = error_calc_SE_k(estimations3, GT_data );
    end
    errors(q,1) = mean(SPEC_error);
    errors(q,2) = mean(CONT_error);
    errors(q,3) = mean(ASAP_error);
end
errors
figure;
hold on;
plot(pValues, errors(:,3),'--k','LineWidth',3);
plot(pValues, errors(:,1),'-.','Color','red','LineWidth',3.3);
plot(pValues, errors(:,2),'Color','blue','LineWidth',3);
legend('Separation','Spectral','Contraction');
fn = sprintf('n = %d, d = %d', n, d);
title(fn);
xlabel('Non-outliers rate');
ylabel('MSE');
hold off;
end

