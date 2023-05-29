% a testing script for the chordal flag mean implementation
addpath '../MatlabCode';

% number of random points on the flag
N = 100;

% unweighted points
weights = ones(N);

flag_type = [1,2,3];

% number of test points
n_test_pts = 100;

% random flag points
flag_pts = zeros(4,3,N);

%sample the points near a center point
rand_mat = rand(4,3);
[U,~,~] = svd(rand_mat);
center_pt = U(:,1:3);

%sample the points
for i = 1:N
    rand_mat = rand(4,3)*.0001 + center_pt;
    [U,~,~] = svd(rand_mat);
    flag_pts(:,:,i) = U(:,1:3);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%the chordal flag mean
flag_mean_est = chordal_flag_mean(flag_pts, weights);

%the objective function value for the chordal flag mean
flag_mean_dist = calculate_chordal_distance(flag_pts, flag_mean_est, flag_type, false);

%the objective values (mean) for random test points
test_dists = zeros(n_test_pts,1);
for i =1:n_test_pts
    [U,~,~] = svd(center_pt + rand(4,3)*.0000001);
    test_pt = U(:,1:3);
    test_dists(i) = calculate_chordal_distance(flag_pts, test_pt, flag_type, false);
end

%flag_mean_dist should be less than test_dists
sum(test_dists < flag_median_dist) %this should be 0

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%the chordal flag median
flag_median_est = chordal_flag_mean(flag_pts, weights);

%the objective function value for the chordal flag median
flag_mean_dist = calculate_chordal_distance(flag_pts, flag_mean_est, flag_type, false);

%the objective values (median) for random test points
test_dists = zeros(n_test_pts,1);
for i =1:n_test_pts
    [U,~,~] = svd(center_pt + rand(4,3)*.0000001);
    test_pt = U(:,1:3);
    test_dists(i) = calculate_chordal_distance(flag_pts, test_pt, flag_type, true);
end

%flag_median_dist should be less than test_dists
sum(test_dists < flag_median_dist)  %this should be 0
