
addpath './Govindu';

N = 1000;
perturbAxis = 0.005;

% random rotation axis
meanAngle = pi/4;
rotationVector = randn(3,1);
rotationVector = rotationVector./norm(rotationVector);

axang = [rotationVector' meanAngle];
gndRotVec = rotationMatrixToVector(axang2rotm(axang));
 
%do for each method
%sample variance for angles - (mean angle)
%sample variance of the axis - (mean axis)
%z is error of estimated average

Rs = cell(N,1);
% now perturb the rotation vector and create many rotations
for i=1:N

    rotationVectorPerturb = rotationVector + perturbAxis*randn(3,1);
    rotationVectorPerturb = rotationVectorPerturb./norm(rotationVectorPerturb);

    % random rotation angles around the axis
    %rotationAngles = mod(randn(N,1), 2*pi)-pi;
    %generate rotation angles between -pi/2 and pi/2
    rotationAngle = randn()/4 + meanAngle;
    
    axang = [rotationVectorPerturb' rotationAngle];
    Rs{i} = (axang2rotm(axang));
end

% now fill in the random translations
Ts = cell(N,1);
for i=1:N
    % I leave T to be 0 for now
    Ts{i} = [Rs{i} [0,0,0]' ; [0 0 0 1]];
end

% generate some weights
Weights = ones(N,1);

Mu = mean_se3_govindu(Ts, Weights, eps);



% Mu_mank_old = zeros(4);
% err = 10;
% while err > .00001
%     Mu_mank = mean_se3_flag(Ts, Weights, 50);
%     err = norm(Mu_mank_old - Mu_mank);
%     Mu_mank_old = Mu_mank;
%     err
% end
% Mu_mank

%%%%%%%%%%%%%%%%%%%%%%
%uncomment to plot errors as a function of lambda

% to find the best lambda
% norm_old = 3;
% norms = zeros(1,200);
% for lambda= 1:200
%     Mu_mank = mean_se3_flag(Ts, Weights, lambda);
%     rotationVectorMu = rotationMatrixToVector(Mu_mank(1:3,1:3));
%     norm_new = norm(rotationVectorMu);
%     norms(lambda) = norm_new;
%     if norm_new < norm_old
%         rotationVectorMu_mank = rotationVectorMu;
%         norm_old = norm_new;
%         best_lambda = lambda;
%     end
% end
% fig1 = figure(2);
% plot(1:200, norms)

%%%%%%%%%%%%%%%%%%%%%%

Mu_mank = mean_se3_flag(Ts, Weights, 50);

% Mu_mank_med = median_se3_flag(Ts, Weights, 50);

Mu_quat = mean_se3_quat_tra(Ts, Weights);

Mu_naive = mean_se3_naive(Ts, Weights);

% Mu should be zero degrees around the axis:
rotationVectorMu = rotationMatrixToVector(Mu(1:3,1:3));

rotationVectorMu_mank = rotationMatrixToVector(Mu_mank(1:3,1:3));

% rotationVectorMu_mank_med = rotationMatrixToVector(Mu_mank_med(1:3,1:3)); 

rotationVectorMu_quat = rotationMatrixToVector(Mu_quat(1:3,1:3));

rotationVectorMu_naive = rotationMatrixToVector(Mu_naive(1:3,1:3));

allSolutions = {rotationVectorMu_naive, rotationVectorMu_quat, rotationVectorMu, rotationVectorMu_mank};
allSolutionNames = {'naive', 'qt', 'govindu', 'mankovich'};

gndRotVec
plot_solutions(gndRotVec, allSolutions, allSolutionNames);


% the following two lines should be roughly equal (up to a sign maybe)
% mean(rotationAngles)
% norm(rotationVectorMu)
% norm(rotationVectorMu_mank)
% % norm(rotationVectorMu_mank_med)
% norm(rotationVectorMu_quat)
% norm(rotationVectorMu_naive)

