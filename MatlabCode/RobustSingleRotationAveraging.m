
%% Example: Average 100 rotations (50 inliers, 50 outliers)

n_inliers = 50;
n_outliers = 30;
inlier_noise_level = 5; %deg;
R_true = RandomRotation(pi); 


% 1. Create input rotaions:
n_samples = n_inliers + n_outliers;
R_samples = cell(1, n_samples);
                
for i = 1:n_samples
    if (i <= n_inliers)
        % Inliers: perturb by 5 deg.
        axis_perturb = rand(3,1)-0.5;
        axis_perturb = axis_perturb/norm(axis_perturb);
        angle_perturb = normrnd(0,inlier_noise_level/180*pi); 
        R_perturb = RotationFromUnitAxisAngle(axis_perturb, angle_perturb);
        R_samples{i} = R_perturb*R_true;
    else
        % Outliers: completely random.
        R_samples{i} = RandomRotation(pi); 
    end   
end

% 2-a. Average them using Hartley's L1 geodesic method 
% (with our initialization and outlier rejection scheme):

b_outlier_rejection = false;
n_iterations = 10;
thr_convergence = 0.001;
tic;
%R_geodesic = GeodesicL1Mean(R_samples, b_outlier_rejection, n_iterations, thr_convergence);
R_chordal_no_outlier = ChordalL1Mean(R_samples, b_outlier_rejection, n_iterations, thr_convergence);
time_geodesic = toc;

% 2-b. Average them using our approximate L1 chordal method 
% (with our initialization and outlier rejection shceme)

b_outlier_rejection = true;
n_iterations = 100;
thr_convergence = 0.001;
tic;
R_chordal = ChordalL1Mean(R_samples, b_outlier_rejection, n_iterations, thr_convergence);
time_chordal = toc;


% 3. Evaluate the rotation error (deg):

error_ChordalMean = abs(acosd((trace(R_true*R_chordal_no_outlier')-1)/2));
error_ChordalL1Median = abs(acosd((trace(R_true*R_chordal')-1)/2));

disp(['Error (chordal L1 mean) = ', num2str(error_ChordalMean), ' deg, took ', num2str(time_geodesic*1000), ' ms'])
disp(['Error (chordal L1 median) = ', num2str(error_ChordalL1Median), ' deg, took ', num2str(time_chordal*1000), ' ms' ])
disp('')


% now try our algorithm
Ts = Rts_to_Ts(R_samples);
Weights = ones(length(Ts),1);
tic();
% R_true
Mu_flag = mean_se3_mankovich(Ts, Weights, 50);
Mu_qt = mean_se3_quat_tra(Ts, Weights);
Mu_govindu = mean_se3_govindu(Ts, Weights, eps);
Mu_naive = mean_se3_naive(Ts, Weights);
R_naive = Mu_naive(1:3,1:3);
R_qt = Mu_qt(1:3,1:3);
R_govindu = Mu_govindu(1:3,1:3);
R_flag = Mu_flag(1:3,1:3);

time_mank = toc();
error_naive = abs(acosd((trace(R_true*R_naive')-1)/2));
error_qt = abs(acosd((trace(R_true*R_qt')-1)/2));
error_govindu = abs(acosd((trace(R_true*R_govindu')-1)/2));
error_flag = abs(acosd((trace(R_true*R_flag')-1)/2));
disp(['Error (naive) = ', num2str(error_naive), ' deg']);
disp(['Error (qt) = ', num2str(error_qt), ' deg']);
disp(['Error (govindu) = ', num2str(error_govindu), ' deg']);
disp(['Error (flag) = ', num2str(error_flag), ' deg']);
