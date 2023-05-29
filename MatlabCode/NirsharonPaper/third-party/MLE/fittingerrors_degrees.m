function errs = fittingerrors_degrees(problem, Rhat)
% Given a synchronization problem structure and an estimator Rhat for the
% rotations, computes the fitting error in degree of each measurement and
% returns them in a vector. This is for rotations in SO(2) or SO(3) only.

    assert(ismember(problem.n, [2 3]), ...
           'This makes sense for n = 2 or 3 only.');

    Ri = Rhat(:, :, problem.I);
    Rj = Rhat(:, :, problem.J);
    H = problem.H;
    Z = multiprod(multitransp(Ri), multiprod(H, Rj));
    errs = zeros(problem.M, 1);
    for k = 1 : length(errs)
        errs(k) = norm(logm(Z(:, :, k)), 'fro')*sqrt(2)*180/pi;
    end

end
