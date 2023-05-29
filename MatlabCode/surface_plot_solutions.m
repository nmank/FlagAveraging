function errsAll = surface_plot_solutions(gndRotVecs, RotVecVars, RotAngVars, solRotations, solNames, show3d)

if (~exist('show3d','var'))
    show3d = 0;
end

n = length(solRotations);
n_trials = length(solRotations{1});

errsAll = cell(n,1);
for i=1:n
    errs = zeros(n_trials,1);

    for j=1:n_trials
        %find the rotation angle
        Rgnd = rotationVectorToMatrix(gndRotVecs{j});
        rotVec = solRotations{i}{j};
        if(sign(rotVec(1))~=sign(gndRotVecs{j}(1)))
            rotVec = - rotVec;
        end
        R = rotationVectorToMatrix(rotVec);
        theta = acos(0.5*(trace ( R * Rgnd')-1));
        errs(j) = theta;

    end
    if (show3d)
        figure(i)
        scatter3(RotVecVars, RotAngVars, errs);
        title(solNames{i});
    end
    errsAll{i} = errs;
end

end