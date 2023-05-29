function [flg_mean] = chordal_flag_mean(data, weights, flag_type, oriented)

    if ~exist('flag_type','var')
        flag_type = [1,2,3];
    end

    if ~exist('oriented','var')
        oriented = true;
    end
    
    [n,k,~] = size(data);
    
    weightsn = repmat(weights', n, 1);


    %generate projections and ys
    ps = {};
    Is = {};
    means = {};
    for i=1:length(flag_type)
        if i == 1
            flag_type_m1 = 1;
        else
            flag_type_m1 = flag_type(i-1)+1;
        end
        %generate projections
        % should this be the square root of the weights? 
        % do the weights have to add up to 1?
        xi = squeeze(data(:,i,:)); 
        pi = weightsn.*xi*xi';
        meani= mean(xi,2);
        means{i} = meani/norm(meani);
        ps{i} = pi;

        %define the I matrices
        I = zeros(flag_type(k));
        I(flag_type_m1:flag_type(i), flag_type_m1:flag_type(i)) = 1;
        Is{i} = I;
    end

%     x1 = squeeze(data(:,1,:));
%     x2 = squeeze(data(:,2,:));
%     x3 = squeeze(data(:,3,:));
% 
%     ps{1} = x1*x1' + x2*x2' + x3*x3';
%     ps{2} = x2*x2' + x3*x3';
%     ps{3} = x3*x3';

%     %generate the problem as an optimization problem on the stiefel manifold 
%     I1 = [1, 0 ,0; 0, 0, 0; 0, 0, 0];
%     I2 = [0, 0 ,0; 0, 1, 0; 0, 0, 0];
%     I3 = [0, 0 ,0; 0, 0, 0; 0, 0, 1];
    
    St = stiefelfactory(n,k);
    problem.M = St;
    
    problem.cost = @cost;
    function [f] = cost(X)     
        f = 0;
        for i = 1:length(flag_type)
            f = f - trace(Is{i} * X' * ps{i} * X );
        end
    end
    
    problem.egrad = @egrad;
    function [G] = egrad(X)
        G = 0;
        for i = 1:length(flag_type)
            G = G - 2*(ps{i} * X * Is{i});
        end
    end

    
    
    %solve the optimization
    options.maxiter = 3000;
    options.tolgradnorm = 1e-7;
    options.verbosity = 0;
    options.debug = 0;
    %options.tolgradnorm = 1e-7;

    %try different initialization! try idenity, naive mean or sota
    %method

    %initialize form euclidean mean projected onto flag manifold!

    %initial point
%     i_p = eye(4);
    mu_data = mean(data, k);
    [Q, ~] = qr(mu_data);
    i_p = Q(:,1:k);


    flg_mean_unoriented = trustregions(problem, i_p(:,1:3), options);

    %make the result live on an oriented flag
    if oriented
        flg_mean = zeros(n,k);
        for i=1:k
            if means{i}'*flg_mean_unoriented(:,i) < 0
                flg_mean(:,i) = -flg_mean_unoriented(:,i);
            else
                flg_mean(:,i) = flg_mean_unoriented(:,i);
            end
        end
    else
        flg_mean = flg_mean_unoriented;
    end

end