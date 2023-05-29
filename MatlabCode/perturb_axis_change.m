
addpath './Govindu';

warning('off', 'manopt:getHessian:approx');

n_experiments = 50;
N = 100;
n_perturb = 60;
perturb_div = 30;
perturbAxes = zeros(n_perturb,1);
curverrs = cell(n_perturb,1);

allSolutionNames = {'Naive', 'QT', 'Govindu', 'Flag (ours)'};

Rs = cell(N,1);
for ii=1:n_perturb
    perturbAxis = ii/perturb_div;
    
    % random rotation axis
    meanAngle = pi/4;
    
    
    gtRotVecs = {};
    RotVecVars = zeros(n_experiments,1);
    RotAngVars = zeros(n_experiments,1);
    
    rotationVectorMu_naives = {};
    rotationVectorMu_quats = {};
    rotationVectorMus = {};
    rotationVectorMu_manks = {};

    angularDifferencesExp = zeros(n_experiments, N);
    
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

        angularDifferences = zeros(1, N);
        % now perturb the rotation vector and create many rotations
        for i=1:N
        
            pRand = perturbAxis*randn(3,1);
            rotationVectorPerturb(:,i) = rotationVector + pRand;
            rotationVectorPerturb(:,i) = rotationVectorPerturb(:,i)./norm(rotationVectorPerturb(:,i));
            
            u = rotationVector;
            v = rotationVectorPerturb(:,i);
            angularDiff = atan2(norm(cross(u,v)),dot(u,v));
            angularDifferences(i) = angularDiff;
    
    %         rotVecDist(i) = abs(rotationVectorPerturb(:,i)' * rotationVector); %cosine of the angle between vectors
            rotVecDist(i) = norm(rotationVectorPerturb(:,i) -  rotationVector);
        
            % random rotation angles around the axis
            %rotationAngles = mod(randn(N,1), 2*pi)-pi;
            %generate rotation angles between -pi/2 and pi/2
            rotationAngle(i) = randn()/4 + meanAngle;
            
            axang = [rotationVectorPerturb(:,i)' rotationAngle(i)];
            Rs{i} = (axang2rotm(axang));
        end
%         disp(rotVecDist)

        angularDifferencesExp(exp_n, :) = angularDifferences;
    % 
        %angle variance
        RotAngVars(exp_n) = var(rotationAngle - meanAngle);
        %norm of rotation variance
        % RotVecVars(exp_n) = var(rotVecDist);
        RotVecVars(exp_n) = var(angularDifferences);
    
        
        % now fill in the random translations
        Ts = cell(N,1);
        for i=1:N
            % I leave T to be 0 for now
            Ts{i} = [Rs{i} [0,0,0]' ; [0 0 0 1]];
        end
        
        % generate some weights
        Weights = ones(N,1);
        
        Mu = mean_se3_govindu(Ts, Weights, eps);
        
        
        Mu_mank = mean_se3_mankovich(Ts, Weights, 50);
        
        % Mu_mank_med = median_se3_mankovich(Ts, Weights, 50);
        
        Mu_quat = mean_se3_quat_tra(Ts, Weights);
        
        Mu_naive = mean_se3_naive(Ts, Weights);
        
        % Mu should be zero degrees around the axis:
        rotationVectorMus{exp_n} = rotationMatrixToVector(Mu(1:3,1:3));
        
        rotationVectorMu_manks{exp_n} = rotationMatrixToVector(Mu_mank(1:3,1:3));
        
        % rotationVectorMu_mank_med = rotationMatrixToVector(Mu_mank_med(1:3,1:3)); 
        
        rotationVectorMu_quats{exp_n} = rotationMatrixToVector(Mu_quat(1:3,1:3));
        
        rotationVectorMu_naives{exp_n} = rotationMatrixToVector(Mu_naive(1:3,1:3));
    
        gtRotVecs{exp_n} = gndRotVec;
    
    end
    
    allSolutions = {rotationVectorMu_naives, rotationVectorMu_quats, rotationVectorMus, rotationVectorMu_manks};
    errsAll = surface_plot_solutions(gtRotVecs, RotVecVars, RotAngVars, allSolutions, allSolutionNames);
    
    curverr = cell(length(errsAll), 1);
    for i=1:length(errsAll)
        curverr{i} = mean(errsAll{i});
    end


    curverrs{ii} = mat2cell(cellfun(@rad2deg, curverr), [1,1,1,1],[1]);

    disp(['exp: ' num2str(ii)] );
    disp(['naive: ' num2str(curverr{1}) ', qt: ' num2str(curverr{2}), ', govindu: ' num2str(curverr{3}) ', mank: ' num2str(curverr{4})]);

end

%% the rest is only for plotting:

ms = 60;
f = figure;
beautify_plot; box on;
clrs = colororder;
x = (1:10)./perturb_div;
y = [];
for i=1:n_perturb
    hold on
    scatter(i/perturb_div, curverrs{i}{1}, ms, clrs(1,:), 'square', 'LineWidth', 2, 'DisplayName', allSolutionNames{1});
    scatter(i/perturb_div, curverrs{i}{2}, ms, clrs(2,:), 'x', 'LineWidth', 2, 'DisplayName', allSolutionNames{2});
    scatter(i/perturb_div, curverrs{i}{3}, ms, clrs(3,:), '>', 'LineWidth', 2, 'DisplayName', allSolutionNames{3});
    scatter(i/perturb_div, curverrs{i}{4}, ms, clrs(4,:), 'LineWidth', 2, 'DisplayName', allSolutionNames{4   });
    y = [y curverrs{i}{1}];
end
legend([allSolutionNames], 'Location', 'southeast');
xlabel('Perturbation (noise) level');
ylabel('Angular error (\circ)');
axis([0 pi/2 0 rad2deg(0.6)]);

% zoom-box position:
pos = [.18 .6 .25 .3];

xbounds = [1/perturb_div 10/perturb_div];

% pos = [1/perturb_div 10/perturb_div curverrs{1}{1} curverrs{10}{1}]; 
p = gca;
% Calculate x,y points of zoomPlot
x1 = (pos(1)-p.Position(1))/p.Position(3)*diff(p.XLim)+p.XLim(1);
x2 = (pos(1)+pos(3)-p.Position(1))/p.Position(3)*diff(p.XLim)+(p.XLim(1));
y1 = (pos(2)-p.Position(2))/p.Position(4)*diff(p.YLim)+p.YLim(1);
y2 = ((pos(2)+pos(4)-p.Position(2))/p.Position(4))*diff(p.YLim)+p.YLim(1);
% Plot lines connecting zoomPlot to original plot points
index = find(x>=xbounds(1) & x<=xbounds(2)); % Find indexes of points in zoomPlot
rectangle('Position',[xbounds(1) min(y(index)) diff(xbounds) max(y(index))-min(y(index))]);
hold on

vertex = [1 3];
if any(vertex==1)
    plot([xbounds(1) x1], [max(y(index)) y2], 'k', 'LineWidth', 1.5,'HandleVisibility','off'); % Line to vertex 1
end
if any(vertex==2)
    plot([xbounds(2) x2], [max(y(index)) y2], 'k', 'LineWidth', 1.5,'HandleVisibility','off'); % Line to vertex 2
end
if any(vertex==3)
    plot([xbounds(2) x2], [min(y(index)) y1], 'k', 'LineWidth', 1.5,'HandleVisibility','off'); % Line to vertex 4
end
if any(vertex==4)
    plot([xbounds(1) x1], [min(y(index)) y1], 'k', 'LineWidth', 1.5,'HandleVisibility','off'); % Line to vertex 3
end

z = axes('position', pos);
box on; % put box around new pair of axes
grid on;
set(z,'yticklabel',[]);

for i=1:10
    hold on
    scatter(i/perturb_div, curverrs{i}{1}, ms, clrs(1,:), 'square', 'LineWidth', 2);
    scatter(i/perturb_div, curverrs{i}{2}, ms, clrs(2,:), 'x', 'LineWidth', 2);
    scatter(i/perturb_div, curverrs{i}{3}, ms, clrs(3,:), '>', 'LineWidth', 2);
    scatter(i/perturb_div, curverrs{i}{4}, ms, clrs(4,:), 'LineWidth', 2);
end
axis tight;
% export_fig('perturb_axis_change.pdf');
