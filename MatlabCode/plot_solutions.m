function [errs] = plot_solutions(gndRotVec, solRotations, solNames)

n = length(solRotations);

Rgnd = rotationVectorToMatrix(gndRotVec);
errs = zeros(n,1);

for i=1:n
    rotVec = solRotations{i};
    if(sign(rotVec(1))~=sign(gndRotVec(1)))
        rotVec = - rotVec;
    end
    R = rotationVectorToMatrix(rotVec);
    theta = acos(0.5*(trace ( R * Rgnd')-1));
    errs(i) = theta;
end

bar(categorical(solNames),  errs);
title('Errors');

end