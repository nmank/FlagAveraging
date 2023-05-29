function [ H ] = MakeAffinityMatrixSO(SO_array, confidence_weights, noise_func, parms, outliers)
% Construct the affinity matrix from SE data, with noise given by the
% function noise_func and its parameters, given by parms
%
% N.S. May 2016

n = size(confidence_weights,1);
d = size(SO_array,1);
if nargin<5
    outliers = 0;
end

if nargin<4
    parms = [];
end

if nargin<3
    noise_func = @(x) eye(d);
end


H = zeros(n*(d));

for l=1:n
    for m=(l+1):n
        ind1 = 1+(l-1)*(d);
        ind2 = 1+(m-1)*(d);
        if confidence_weights(l,m)
            current_measurement = SO_array(:,:,l)*noise_func(parms)*SO_array(:,:,m)';
            H(ind1:(ind1+d-1),ind2:(ind2+d-1))= current_measurement;
        else   % missing data case / outlier
            if outliers
                [q, ~] = qr(randn(d));
                current_measurement = q^2; 
            else
                current_measurement = zeros(d);
            end
            H(ind1:(ind1+d-1),ind2:(ind2+d-1))= current_measurement;
        end
    end
end
H = H + H' + eye(n*(d));

end

