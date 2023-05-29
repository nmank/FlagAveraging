
addpath './Govindu';

warning('off', 'manopt:getHessian:approx');

n_experiments = 500;
N = 100;
perturbAxis = 0.5;

% random rotation axis
meanAngle = pi/4;


gtRotVecs = {};
RotVecVars = zeros(n_experiments,1);
RotAngVars = zeros(n_experiments,1);

rotationVectorMu_naives = {};
rotationVectorMu_quats = {};
rotationVectorMus = {};
rotationVectorMu_manks = {};

for exp_n=1:n_experiments
    rotationVector = randn(3,1);
    rotationVector = rotationVector./norm(rotationVector);
    
    axang = [rotationVector' meanAngle];
    
    gndRotVec = rotationMatrixToVector(axang2rotm(axang));
     
    %do for each method
    %sample variance for angles - (mean angle)
    %sample variance of the axis - (mean axis)
    %z is error of estimated average
    rotationAngle = zeros(1,N);
    rotationVectorPerturb = zeros(3,N);
    
    Rs = cell(N,1);
    rotVecDist = zeros(N,1);
    % now perturb the rotation vector and create many rotations
    for i=1:N
    
        rotationVectorPerturb = rotationVector + perturbAxis*randn(3,1);
        rotationVectorPerturb(:,i) = rotationVectorPerturb./norm(rotationVectorPerturb);

%         rotVecDist(i) = abs(rotationVectorPerturb(:,i)' * rotationVector); %cosine of the angle between vectors
        rotVecDist(i) = norm(rotationVectorPerturb(:,i) -  rotationVector);
    
        % random rotation angles around the axis
        %rotationAngles = mod(randn(N,1), 2*pi)-pi;
        %generate rotation angles between -pi/2 and pi/2
        rotationAngle(i) = randn()/4 + meanAngle;
        
        axang = [rotationVectorPerturb(:,i)' rotationAngle(i)];
        Rs{i} = (axang2rotm(axang));
    end
    % disp(rotVecDist)
% 
    %angle variance
    RotAngVars(exp_n) = var(rotationAngle - meanAngle);
    %norm of rotation variance
    RotVecVars(exp_n) = var(rotVecDist);

    %angle mean
%     RotAngVars(exp_n) = mean(rotationAngle - meanAngle);
    %norm of rotation mean
%     RotVecVars(exp_n) = mean(rotVecDist);


%     RotVecVars(exp_n) = norm(var(rotationVectorPerturb,0,2));
%     RotVecVar = var(rotationVectorPerturb,0,2);
%     RotVecVars(exp_n) =  RotVecVar(1);
    
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
    %     Mu_mank = mean_se3_mankovich(Ts, Weights, 50);
    %     err = norm(Mu_mank_old - Mu_mank);
    %     Mu_mank_old = Mu_mank;
    %     err
    % end
    % Mu_mank
    
    
    % to find the best lambda
    % norm_old = 3;
    % norms = zeros(1,200);
    % for lambda= 1:200
    %     Mu_mank = mean_se3_mankovich(Ts, Weights, lambda);
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
    % plot(lambda, norms)
    
    Mu_mank = mean_se3_flag(Ts, Weights, 50);
    
    % Mu_mank_med = median_se3_flag(Ts, Weights, 50);
    
    Mu_quat = mean_se3_quat_tra(Ts, Weights);
    
    Mu_naive = mean_se3_naive(Ts, Weights);
    
    % Mu should be zero degrees around the axis:
    rotationVectorMus{exp_n} = rotationMatrixToVector(Mu(1:3,1:3));
    
    rotationVectorMu_manks{exp_n} = rotationMatrixToVector(Mu_mank(1:3,1:3));
    
    % rotationVectorMu_mank_med = rotationMatrixToVector(Mu_mank_med(1:3,1:3)); 
    
    rotationVectorMu_quats{exp_n} = rotationMatrixToVector(Mu_quat(1:3,1:3));
    
    rotationVectorMu_naives{exp_n} = rotationMatrixToVector(Mu_naive(1:3,1:3));

    gtRotVecs{exp_n} = gndRotVec;

    % the following two lines should be roughly equal (up to a sign maybe)
    % mean(rotationAngles)
    % norm(rotationVectorMu)
    % norm(rotationVectorMu_mank)
    % % norm(rotationVectorMu_mank_med)
    % norm(rotationVectorMu_quat)
    % norm(rotationVectorMu_naive)

    disp(['exp: ' num2str(exp_n)] );
end

allSolutions = {rotationVectorMu_naives, rotationVectorMu_quats, rotationVectorMus, rotationVectorMu_manks};
allSolutionNames = {'naive', 'qt', 'govindu', 'mankovich'};
errsAll = surface_plot_solutions(gtRotVecs, RotVecVars, RotAngVars, allSolutions, allSolutionNames);
surface_plot_solutions2(gtRotVecs, RotVecVars, RotAngVars, allSolutions);

figure; beautify_plot;
for i=1:length(errsAll)
    curerr = mean(errsAll{i});
    curvar = var(errsAll{i});
    hold on, bar(i, curerr);
end
legend(allSolutionNames);
for i=1:length(errsAll)
    curerr = mean(errsAll{i});
    curvar = var(errsAll{i});
    hold on;
    er = errorbar(i,curerr,-curvar,curvar);
    er.Color = [0 0 0];
    er.LineStyle = 'none';
    er.LineWidth = 2;
    %hold off
end