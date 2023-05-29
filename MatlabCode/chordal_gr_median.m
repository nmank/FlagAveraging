function [flg_median] = chordal_gr_median(data, weights)


%parameter to avoid dividing by 0 in weights
eps = .0001;

[n,k,~] = size(data);

%always initialize from the same point (for now...)
rng('default');
rng(1);
[Q,~] = qr(rand(n,k));

flg_median = Q(:,1:k);

% errMean = 1;
err = 1;
it = 1;

while it<20
    med_weights = 1.0./max(err, eps);

    Weights = med_weights.*weights;
    %Weights = Weights./norm(Weights);

    flg_median = chordal_gr_mean(data, Weights);
    
    err = chordal_distance(data, flg_median, k);
    it = it + 1;
    % mean(err)
end
