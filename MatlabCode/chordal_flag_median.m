function [flg_median] = chordal_flag_median(data, weights, flag_type, oriented)

if ~exist('flag_type','var')
    flag_type = [1,2,3];
end

if ~exist('oriented','var')
    oriented = true;
end

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

    flg_median = chordal_flag_mean(data, Weights, flag_type, oriented);
    
    err = chordal_distance(data, flg_median, flag_type);
    it = it + 1;
    % mean(err)
end



% def real_flag_median_qr(data, flag_type, n_iters = 100):
%     n = data[0].shape[0]
%     Y = np.linalg.qr(np.random.rand(n, n))[0]
% 
%     errs = []
%     errs.append(real_flag_mean_objective(data, Y, flag_type, median = True))
%     for i in range(n_iters):
% 
%         weights = [calc_weight(d,Y,flag_type) for d in data]
%         Y = real_flag_mean_qr(data, flag_type, weights)
%         errs.append(real_flag_mean_objective(data, Y, flag_type, median = True))
%     plt.plot(errs)
%     return Y