function c = langevinnormalizations(n, kappa, scaling)
% Calls langevinnormalization once for each different kappa.
%
% function c = langevinnormalizations(n, kappa)
%
% Uses 'unique' to avoid redundant computations, since often the input
% vector kappa will contain the same value mor than once. For example, if
% the vector kappa is constant, langevinnormalization is called only once.
% The output vector c always has the same size as the input vector kappa.
%
% Nicolas Boumal, UCLouvain, Jan. 16, 2013.

    assert(numel(n) == 1);
    
    if ~exist('scaling', 'var') || isempty(scaling)
        scaling = 0;
    end
    assert(scaling == 0 || scaling == 1);
    
    % Identify the different values of kappa.
    [~, I, J] = unique(kappa(:), 'rows');
    
    % For each such value, call langevinnormalization once.
    c = zeros(size(I));
    for k = 1 : numel(c)
        c(k) = langevinnormalization(n, kappa(I(k)), scaling);
    end
    
    % Assign the normalization coeffs to the appropriate entries of c so
    % that c has the same size as the input and each entry of c
    % corresponds to the langevinnormalization for the corresponding kappa.
    c = c(J);
    c = reshape(c, size(kappa));
    
end
