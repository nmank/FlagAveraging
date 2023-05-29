function [Mu] = median_se3_mankovich(Ts, Weights, lambda)

N = length(Ts);

if (~exist('Weights','var'))
    Weights = ones(N,1);
end

if (~exist('lambda','var'))
    lambda = 20;
end

flag_points = zeros(4,3,N);

for i=1:N
    pt = Ts{i};
    mu = pt(1:3,1:3);
    b = pt(1:3,4);
    
    so_pt = se_to_so(mu, b, lambda);

    flag_points(:,:,i) = so_to_flag(so_pt);
end

flag_mean = chordal_flag_median(flag_points, Weights);
%flag_mean = chordal_flag_IRLS(flag_points, Weights);
%flag_mean = chordal_gr_mean(flag_points, Weights)

% why do I need a transpose here?
% flag_mean(1:3,1:3)=flag_mean(1:3,1:3)';

so_mean = flag_to_so(flag_mean);

[se_mean_mu, so_mean_b] = so_to_se(so_mean, lambda);

Mu = [se_mean_mu so_mean_b ; [0 0 0 1]];





