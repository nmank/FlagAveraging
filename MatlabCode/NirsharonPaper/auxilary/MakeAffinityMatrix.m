function [ H, Avg_SNR ] = MakeAffinityMatrix(SEk_array, confidence_weights, noise_func, parms, outliers, outliers_parm)
% Construct the affinity matrix from SE data, with noise given by the
% function noise_func and its parameters, given by parms
%
% N.S. May 2016

n = size(confidence_weights,1);
d = size(SEk_array,1)-1;

if nargin<6
    outliers_parm = [];
end

if nargin<5
    outliers = 0;
end

if nargin<4
    parms = [];
end

if nargin<3
    noise_func = @(x) eye(d+1);
end


H = zeros(n*(d+1));
Avg_SNR = 0;
for l=1:n
    for m=(l+1):n
        ind1 = 1+(l-1)*(d+1);
        ind2 = 1+(m-1)*(d+1);
        if confidence_weights(l,m)
            N = noise_func(parms);
            logmN = logm(N);
            logmM = logm(SEk_array(:,:,l)*inverse_SE_k(SEk_array(:,:,m)));
            %if logmN~=0
                Avg_SNR = Avg_SNR + snr(logmM(:),logmN(:));
            %else
            %    Avg_SNR = inf;
            %end
            SE_measurement = SEk_array(:,:,l)*N*inverse_SE_k(SEk_array(:,:,m));
            H(ind1:(ind1+d),ind2:(ind2+d)) = SE_measurement;
            H(ind2:(ind2+d),ind1:(ind1+d)) = inverse_SE_k(SE_measurement);
        else   % missing data case / outlier
            if outliers
                SE_measurement = make_random_SE_k(d); %to be eddited
                inv_SE_m = inverse_SE_k(SE_measurement);
            else
                SE_measurement = zeros(d+1); 
                inv_SE_m = zeros(d+1);
            end
            H(ind1:(ind1+d),ind2:(ind2+d)) = SE_measurement;
            H(ind2:(ind2+d),ind1:(ind1+d)) = inv_SE_m; 
        end
    end
end
H = H + eye(n*(d+1));
Avg_SNR = Avg_SNR/nnz(triu(confidence_weights,1));
end

