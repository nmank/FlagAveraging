function w = infoweight(n, kappa1, kappa2, p)
% Computes the information weight associated to a mixture of two isotropic
% Langevin distributions with same mean, concentration kappa1 weighted by
% p and concentration kappa2 weighted by 1-p.
%
% function w = infoweight(n, kappa1, kappa2, p)
%
% See the associate paper:
% http://imaiai.oxfordjournals.org/content/early/2013/09/23/imaiai.iat006.abstract
% N. Boumal, A. Singer, P.-A. Absil and V. D. Blondel,
% Cramér-Rao bounds for synchronization of rotations,
% in Information and Inference, a journal of the IMA, 2013.
%
% See also Nicolas Boumal's PhD thesis for explicit derivations of the
% formulas implemented in this function.
%
% See also: infoweights langevinnormalization
%
% Nicolas Boumal, UCLouvain, Jan. 16, 2013.

    assert(numel(n) == 1);
    assert(numel(kappa1) == 1 && kappa1 >= 0);
    assert(numel(kappa2) == 1 && kappa2 >= 0);
    assert(numel(p) == 1 && p >= 0 && p <= 1);
    
    % Scaling = 1 scales down the langevin normalization constants by a
    % factor of exp(-n*kappa), which is the right order of magnitude we
    % need to balance in the integrands defined below.
    scaling = 1;
    c1 = langevinnormalization(n, kappa1, scaling);
    c2 = langevinnormalization(n, kappa2, scaling);
    
    switch n
        case 2
            w = quad(@integrand2, 0, pi) * (2/pi);
            
        case 3
            w = quad(@integrand3, 0, pi) * (2/pi);
            
        otherwise
            error('infoweight only supports n = 2 and n = 3.');
    end

    
    function h = integrand2(theta)
        ell1 = exp(2*kappa1*(cos(theta)-1)) / c1;
        ell2 = exp(2*kappa2*(cos(theta)-1)) / c2;
        num = (p*kappa1*ell1 + (1-p)*kappa2*ell2).^2;
        den = p*ell1 + (1-p)*ell2;
        h = (num./den) .* sin(theta).^2;
        % This line to prevent num/den to be a 0/0 NaN.
        % When this happens, the actual value is 0, as
        % the numerator decreases much faster than the
        % denominator with theta going to pi.
        h(isnan(h)) = 0;
    end

    function h = integrand3(theta)
        ell1 = exp(kappa1*(1+2*cos(theta)-3)) / c1;
        ell2 = exp(kappa2*(1+2*cos(theta)-3)) / c2;
        num = (p*kappa1*ell1 + (1-p)*kappa2*ell2).^2;
        den = p*ell1 + (1-p)*ell2;
        h = (num./den) .* sin(theta).^2 .* (1-cos(theta));
        h(isnan(h)) = 0;
    end
    
end
