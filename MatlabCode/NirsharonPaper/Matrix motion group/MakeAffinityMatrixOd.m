function [ H, Avg_SNR ] = MakeAffinityMatrixOd(Od_array, W, noise_func, parms, outliers, outliers_parm)
% 
% Construct the affinity matrix from SE data, with noise given by the
% function noise_func and its parameters, given by parms
%
% N.S. June 2017

n = size(W,1);
d = size(Od_array,1);

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
    noise_func = @(x) eye(d);
end


H = zeros(n*d);
Avg_SNR = 0;
for l=1:n
    for m=(l+1):n
        ind1 = 1+(l-1)*(d);
        ind2 = 1+(m-1)*(d);
        if W(l,m)
            N = noise_func(parms);
            logmN = logm(norm(N,'fro'));
            logmM = logm(norm(Od_array(:,:,l)*Od_array(:,:,m)','fro'));
            if logmN~=0
                Avg_SNR = Avg_SNR + snr(logmM(:),logmN(:));
            else
                Avg_SNR = inf;
            end
            Od_measurement = Od_array(:,:,l)*N*Od_array(:,:,m)';
            H(ind1:(ind1+d-1),ind2:(ind2+d-1)) = Od_measurement;
            H(ind2:(ind2+d-1),ind1:(ind1+d-1)) = Od_measurement';
        else   % missing data case / outlier
            if outliers
                Od_measurement = squeeze(make_data_O_d(d,1));
            else
                Od_measurement = zeros(d+1); 
            end
            H(ind1:(ind1+d),ind2:(ind2+d)) = Od_measurement;
            H(ind2:(ind2+d),ind1:(ind1+d)) = Od_measurement'; 
        end
    end
end
H = H + eye(n*d);
Avg_SNR = Avg_SNR/nnz(triu(W,1));
end

