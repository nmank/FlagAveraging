function [flg_median] = chordal_flag_IRLS(data, weights)

eps = .001;

% [n,k,~] = size(data);

% [Q,~] = qr(rand(n,k));

% flg_median = Q(:,1:k);
% flg_median = chordal_flag_mean(data, weights);
% residual_norms = chordal_distance(data, flg_median);

% residual_norms = ones(length(err),1);
it = 1;

weightFun      = @(r) (abs(r) < 1) .* (1 - r.^2).^2;
tuningConstant = 4.685;
Weights = weights;

while it<50
    % med_weights = 1.0./sqrt(residual_norms);
    % thr = compute_mad_thr(residual_norms);
    % ind = (residual_norms > thr);
    % med_weights(ind) = 0;
    %Weights = med_weights.*weights;
    % Weights = med_weights.*weights;
    %Weights = Weights./norm(Weights);

    flg_median = chordal_flag_mean(data, Weights);
    residual_norms = chordal_distance(data, flg_median);
    
    residualLeverages = leverage(residual_norms);
    robustVar = mad(residual_norms, 1);
    r = residual_norms ./ (tuningConstant * robustVar .* sqrt(1 - residualLeverages));
    Weights = weightFun(r).*weights;

    it = it + 1;
    % mean(residual_norms)
end

end


function [thr] = compute_mad_thr(residual_norms)

n_samples = length(residual_norms);

sorted_res_norms = sort(residual_norms);
v_norm_firstQ = sorted_res_norms(ceil(n_samples/4));

med = median(residual_norms);


if (n_samples <= 50)
    thr = max(v_norm_firstQ, 0.5);
    % 2*sqrt(2)*sin(1/2) is approximately 1.356
else
    % thr = max(v_norm_firstQ, 0.25);
    thr = v_norm_firstQ;
    % 2*sqrt(2)*sin(0.5/2) is approximately 0.7
end
end