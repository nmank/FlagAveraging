function errs = surface_plot_solutions2(gndRotVecs, RotVecVars, RotAngVars, solRotations)

n = length(solRotations);
n_trials = length(solRotations{1});
all_errs = zeros(n_trials,1);

for j = 1:n_trials
    Rgnd = rotationVectorToMatrix(gndRotVecs{j});
    errs = zeros(n,1);
    for i = 1:n
        rotVec = solRotations{i}{j};
        if(sign(rotVec(1))~=sign(gndRotVecs{j}(1)))
            rotVec = - rotVec;
        end
        R = rotationVectorToMatrix(rotVec);
        theta = acos(0.5*(trace ( R * Rgnd')-1));
        errs(i) = theta;
    end


    if sum(errs(1:n-1) > errs(n)) < 3
        all_errs(j) = -1;
    else
        all_errs(j) = 1;
    end
      

end

figure(1)
scatter3(RotVecVars, RotAngVars, all_errs);
title('only the goods');

end