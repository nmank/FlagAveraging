% script name: "test_SyncMMG_cleaner"
%
% Run the sync over MMG procedures
%
% N.S, July 2017

clear;

n = 60;
d = 4; 
l = 3;

%----- synthetic data ------
MMG_array = make_data_MMG(d, l, n);

% choose scenario
clean_scenario = 0;
noise_scenario = 1;
noise_missing_scenario = 0;
noise_outliers_scenario = 0;

%----- scenario 1: no noise ----
if clean_scenario
    W = ones(n);
    H = MakeAffinityMatrixMMG(MMG_array, W);
    
    
    estimations_sep = syncMMG_Separation(H, W);
    err_sep = error_calc_MMG(estimations_sep, MMG_array)
    
    lambda = 50;
    
    estimations = SyncMMG_contraction(H, W, lambda);
    [err_cont, shift_cont] = error_calc_MMG(estimations, MMG_array);
    
    err_cont
end

%----- scenario 2: some noise ----
if noise_scenario
    noise_func = @(parm) { make_O_noise(d, parm.sig1) ,  make_O_noise(l, parm.sig2) , rand(d,l)*parm.sig3};
    dif_sig = .05:.1:.6;
    noise_levels = numel(dif_sig);
    W = ones(n);
    err_cont = zeros(noise_levels,1);
    err_sep = zeros(noise_levels,1);
    for k=1:noise_levels
        parm.sig1 = dif_sig(k);
        parm.sig2 = dif_sig(k);%0;
        parm.sig3 = dif_sig(k);
        H = MakeAffinityMatrixMMG(MMG_array, W, noise_func, parm); 

        estimations_sep = syncMMG_Separation(H, W);
        err_sep(k) = error_calc_MMG(estimations_sep, MMG_array);
    
        lambda = 50;    
        estimations = SyncMMG_contraction(H, W, lambda);
        err_cont(k) = error_calc_MMG(estimations, MMG_array);
    end
plot(dif_sig,err_cont,'LineWidth',3);
hold on
plot(dif_sig,err_sep,'--K','LineWidth',3.5);
xlabel('Noise variance');
ylabel('MSE');
set(gca,'FontSize',24)
legend('Contraction','Separation','Location','best')
xlim([dif_sig(1),dif_sig(end)])
nameit = ['noisy_MMGof_d_', num2str(d), '_l_',num2str(l),'_n_', num2str(n),'V1'];
saveas(gcf,nameit,'fig');        

end

%----- scenario 3: some noise, some missing data ----
if noise_missing_scenario
    noise_func = @(parm) { make_O_noise(d, parm.sig1) ,  make_O_noise(l, parm.sig2) , rand(d,l)*parm.sig3};
    fixed_noise = .25;
    availablity = .3:(-.05):.1;
    m = n*(n-1)/2;               % full graph

    availablity_levels = numel(availablity);
    err_cont = zeros(availablity_levels,1);
    err_sep = zeros(availablity_levels,1);
    for k=1:availablity_levels
        p = availablity(k);
        current_avail = floor(p*m);   % number of nonoutliers
        y = randsample(m,current_avail);
        [i,j] = find(triu(ones(n),1));  % indices of upper side blocks
        I = i(y); J=j(y);
        prob_arr = sparse(I,J,ones(numel(I),1),n,n);
        W = eye(n)+prob_arr+prob_arr';
        
        parm.sig1 = fixed_noise;
        parm.sig2 = fixed_noise;
        parm.sig3 = fixed_noise;
        H = MakeAffinityMatrixMMG(MMG_array, W, noise_func, parm); 
        
        lambda = 50;    
        estimations = SyncMMG_contraction(H, W, lambda);
        err_cont(k) = error_calc_MMG(estimations, MMG_array);

        estimations_sep = syncMMG_Separation(H, W);
        err_sep(k) = error_calc_MMG(estimations_sep, MMG_array);
    
    end
plot(availablity,err_cont,'LineWidth',3);
hold on
plot(availablity,err_sep,'--K','LineWidth',3.5);
xlabel('Available data');
ylabel('MSE');
set(gca,'FontSize',24)
legend('Contraction','Separation','Location','best')
xlim([availablity(end),availablity(1)])
nameit = ['missing_data_MMGof_d_', num2str(d), '_l_',num2str(l),'_n_', num2str(n)];
saveas(gcf,nameit,'fig');        

end


%----- scenario 4: some noise, some outliers ----
if noise_outliers_scenario
    noise_func = @(parm) { make_O_noise(d, parm.sig1) ,  make_O_noise(l, parm.sig2) , rand(d,l)*parm.sig3};
    fixed_noise = .15;
    availablity = .5:(.05):.85; % precentage of nonoutliers
    m = n*(n-1)/2;               % full graph

    availablity_levels = numel(availablity);
    err_cont = zeros(availablity_levels,1);
    err_sep = zeros(availablity_levels,1);
    for k=1:availablity_levels
        p = availablity(k);
        current_avail = floor(p*m);   % number of nonoutliers
        y = randsample(m,current_avail);
        [i,j] = find(triu(ones(n),1));  % indices of upper side blocks
        I = i(y); J=j(y);
        prob_arr = sparse(I,J,ones(numel(I),1),n,n);
        W = eye(n)+prob_arr+prob_arr';
        
        parm.sig1 = fixed_noise;
        parm.sig2 = fixed_noise;
        parm.sig3 = fixed_noise;
        H = MakeAffinityMatrixMMG(MMG_array, W, noise_func, parm, 1); 
        
        lambda = 50;    
        estimations = SyncMMG_contraction(H, W, lambda);
        err_cont(k) = error_calc_MMG(estimations, MMG_array);

        estimations_sep = syncMMG_Separation(H, W);
        err_sep(k) = error_calc_MMG(estimations_sep, MMG_array);
    
    end
plot(availablity,err_cont,'LineWidth',3);
hold on
plot(availablity,err_sep,'--K','LineWidth',3.5);
xlabel('Non-outliers rate');
ylabel('MSE');
xlim([availablity(1),availablity(end)])
legend('Contraction','Separation','Location','best')
set(gca,'FontSize',24)
nameit = ['OutliersWnoise_MMGof_d_', num2str(d), '_l_',num2str(l),'_n_', num2str(n)];
saveas(gcf,nameit,'fig');        

end



