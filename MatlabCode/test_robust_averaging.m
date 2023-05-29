
%% Example: Average 100 rotations (50 inliers, 50 outliers)

n_samples = 250;
inlier_noise_level = 5; %deg;
outlier_ratios = [0.025 0.05 0.1 0.2 0.35 0.5 0.7 0.8];
n_experiments = 50;

% parameters for traditional chordal/geodesic robust averages
n_iterations = 100;
thr_convergence = 0.001;

% note, chordal/geodesic-L1 methods are from: 
allMethods = {'Naive', 'QT', 'Govindu', 'Chordal-L_1', 'Geodesic-L_1', 'Chordal-L_1-IRLS', 'Geodesic-L_1-IRLS', 'Flag-L_2 (ours)', 'Flag-IRLS (ours)'};

numOutlierTests = length(outlier_ratios);
numMethods = length(allMethods);
allErrors = zeros(numOutlierTests, numMethods);

for oi = 1:numOutlierTests

    out_ratio = outlier_ratios(oi);

    n_outliers = fix(n_samples*out_ratio);
    n_inliers = n_samples - n_outliers;

    errorsExp = zeros(n_experiments, numMethods);

    for exp_n=1:n_experiments

        R_true = RandomRotation(pi);

        % 1. Create input rotaions:
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

        % outlier rejection is false        
        R_geodesic_no_outlier = GeodesicL1Mean(R_samples, false, n_iterations, thr_convergence);
        R_chordal_no_outlier = ChordalL1Mean(R_samples, false, n_iterations, thr_convergence);

        % 2-b. Average them using our approximate L1 chordal method
        % (with our initialization and outlier rejection shceme)

        % outlier rejection is true
        R_geodesic = GeodesicL1Mean(R_samples, true, n_iterations, thr_convergence);
        R_chordal = ChordalL1Mean(R_samples, true, n_iterations, thr_convergence);

        % 3. Evaluate the rotation error (deg):
        error_ChordalMean = abs(acosd((trace(R_true*R_chordal_no_outlier')-1)/2));
        error_ChordalL1Median = abs(acosd((trace(R_true*R_chordal')-1)/2));
        error_GeodesicMean = abs(acosd((trace(R_true*R_geodesic_no_outlier')-1)/2));
        error_GeodesicL1Median = abs(acosd((trace(R_true*R_geodesic')-1)/2));

        % now try our algorithm as well as others
        Ts = Rts_to_Ts(R_samples);
        Weights = ones(length(Ts),1);
        % R_true
        Mu_flag = mean_se3_flag(Ts, Weights, 50);
        Mu_flag_median = median_se3_flag(Ts, Weights, 50);
        Mu_qt = mean_se3_quat_tra(Ts, Weights);
        Mu_govindu = mean_se3_govindu(Ts, Weights, eps);
        Mu_naive = mean_se3_naive(Ts, Weights);
        R_naive = Mu_naive(1:3,1:3);
        R_qt = Mu_qt(1:3,1:3);
        R_govindu = Mu_govindu(1:3,1:3);
        R_flag = Mu_flag(1:3,1:3);
        R_flag_median = Mu_flag_median(1:3,1:3);

        error_naive = abs(acosd((trace(R_true*R_naive')-1)/2));
        error_qt = abs(acosd((trace(R_true*R_qt')-1)/2));
        error_govindu = abs(acosd((trace(R_true*R_govindu')-1)/2));
        error_flag = abs(acosd((trace(R_true*R_flag')-1)/2));
        error_flag_median = abs(acosd((trace(R_true*R_flag_median')-1)/2));
        disp(['Error (naive) = ', num2str(error_naive), ' deg']);
        disp(['Error (qt) = ', num2str(error_qt), ' deg']);
        disp(['Error (govindu) = ', num2str(error_govindu), ' deg']);
        disp(['Error (chordal mean) = ', num2str(error_ChordalMean), ' deg'])
        disp(['Error (chordal L1 median) = ', num2str(error_ChordalL1Median), ' deg'])
        disp(['Error (geodesic mean) = ', num2str(error_GeodesicMean), ' deg'])
        disp(['Error (geodesic L1 median) = ', num2str(error_GeodesicL1Median), ' deg'])
        disp(['Error (flag) = ', num2str(error_flag), ' deg']);
        disp(['Error (flag-median) = ', num2str(error_flag_median), ' deg']);
        disp(' ')

        errorsExp(exp_n, 1) = error_naive;
        errorsExp(exp_n, 2) = error_qt;
        errorsExp(exp_n, 3) = error_govindu;
        errorsExp(exp_n, 4) = error_ChordalMean;
        errorsExp(exp_n, 5) = error_GeodesicMean;
        errorsExp(exp_n, 6) = error_ChordalL1Median;
        errorsExp(exp_n, 7) = error_GeodesicL1Median;
        errorsExp(exp_n, 8) = error_flag;
        errorsExp(exp_n, 9) = error_flag_median;

    end

    allErrors(oi, :) = mean(errorsExp,1);
end

%% now plot

ms = 60;
f = figure;
beautify_plot; 
box on;
% I don't plot the 'naive'
for i=2:numMethods
    if(i>=8)
        hold on; plot(outlier_ratios, allErrors(:,i),'Marker','o', 'LineWidth', 2, 'DisplayName', allMethods{i});
    else
        hold on; plot(outlier_ratios, allErrors(:,i),'LineWidth', 2, 'DisplayName', allMethods{i});
    end
end
legend([allMethods(2:numMethods)], 'Location', 'northwest');
xlabel('Outlier ratio');
ylabel('Angular error (\circ)');
axis([0 0.82 0 60]);
