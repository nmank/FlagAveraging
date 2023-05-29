function [ H ] = MakeAffinityMatrixMMG(MMG_array, W, noise_func, parms, outliers, outliers_parm)
% 
% Construct the affinity matrix, with noise given by the
% function noise_func and its parameters, given by parms
%
% N.S. June 2017

n = size(W,1);
d = size(MMG_array{1,1},1);
l = size(MMG_array{1,2},1);

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
    noise_func = @(x) {eye(d) , eye(l), zeros(d,l)};
end

% main loop
H = cell(n,n);
for j=1:n
    for m=(j+1):n
        if W(j,m)
            N = noise_func(parms);
            g1 = MMG_action(MMG_array(j,:),N); 
            g2 = MMG_inv(MMG_array(m,:));
            CurrentMeasure = MMG_action(g1, g2);
        else   % missing data case / outlier
            if outliers
                CurrentMeasure = make_data_MMG(d, l, 1);
            else
                CurrentMeasure = []; 
            end
        end
        % summary
        if ~isempty(CurrentMeasure)
            H{j,m} = CurrentMeasure;
            H{m,j} = MMG_inv(H{j,m});
        end

    end
end
for j=1:n
    H{j,j} = {eye(d) , eye(l), zeros(d,l)};
end

end

