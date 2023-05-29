function beta = langevinphasetransition(n, kappa)
% Phase transition coefficient for the eigenvector method under an
% isotropic Langevin noise distribution on SO(n), with concentration
% parameters kappa >= 0.
%
% Returns a vector beta of the same length as kappa such that for all k,
% 
%  beta(k) = beta_n(kappa(k)) = (1/n)*expectation(trace(Z)),
%
% where Z has the aforementioned Langevin distribution.
% In particular, beta_n(0) = 0 and beta_n(+inf) = 1 for all n.
%
% The meaning of this number is detailed in Nicolas Boumal's PhD thesis.
%
% This implementation is valid for n = 2 or 3.
%
% Nicolas Boumal, UCLouvain, Oct. 7, 2013.

    assert(numel(n) == 1);

    I0 = besseli(0, 2*kappa, 1);
    I1 = besseli(1, 2*kappa, 1);
            
    switch n
        
        case 2
            beta = I1./I0;
            
        case 3
            I2 = besseli(2, 2*kappa, 1);
            beta = (I1-I2)./(I0-I1)/3;
            
        otherwise
            error('Invalid value for n. Should be 2 or 3.');
        
    end

end
