function Z = randlangevinmixture(n, kappa1, kappa2, p)
% Generate random rotations with mixed Langevin distributions.
%
% kappa1, kappa2 and p are vectors of same length N.
% Entries in kappa1 and kappa2 are nonnegative.
% Entries in p lie between 0 and 1.
%
% Z is an nxnxN matrix such that each slice Z(:, :, i) is a rotation matrix
% sampled with probability p(i) from a Langevin with concentration
% kappa1(i) and with probability p(i) from a Langevin with concentration
% kappa2(i). The Langevin's are isotropic with mean at the identity.
%
% See also: randlangevin
%
% Nicolas Boumal, UCLouvain, Jan. 17, 2013.

    kappa1 = kappa1(:);
    kappa2 = kappa2(:);
    p = p(:);
    
    N = length(p);
    assert(length(kappa1) == N);
    assert(length(kappa2) == N);

    coin = rand(N, 1);
    mask = coin <= p;
    
    kappa = kappa2;
    kappa(mask) = kappa1(mask);
    
    Z = randlangevin(n, kappa);

end
