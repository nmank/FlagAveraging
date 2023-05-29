function w = infoweights(n, kappa1, kappa2, p)
% Computes infoweight for inputs kappa1, kappa2 & p vectors of same length.
%
% function w = infoweights(n, kappa1, kappa2, p)
%
% Uses 'unique' to avoid redundant computations, since often the triples
% (kappa1, kappa2, p) appear multiple times. For example, if the vectors
% kappa1, kappa2 and p are constant, infoweight is called only once. The
% output vector w always has the same size as the input vectors.
%
% See the associate paper:
% http://imaiai.oxfordjournals.org/content/early/2013/09/23/imaiai.iat006.abstract
% N. Boumal, A. Singer, P.-A. Absil and V. D. Blondel,
% Cramér-Rao bounds for synchronization of rotations,
% in Information and Inference, a journal of the IMA, 2013.
%
% See also: infoweight
%
% Nicolas Boumal, UCLouvain, Jan. 16, 2013.

    assert(numel(n) == 1);
    
    sz = size(kappa1);
    assert(all(size(kappa2) == sz) && all(size(p) == sz));
    
    % Identify the different triplets (kappa1, kappa2, p)
    [~, I, J] = unique([kappa1(:) kappa2(:) p(:)], 'rows');
    
    % For each such triplet, call infoweight once.
    w = zeros(size(I));
    for k = 1 : numel(w)
        w(k) = infoweight(n, kappa1(I(k)), kappa2(I(k)), p(I(k)));
    end
    
    % Assign the information weights to the appropriate entries of w so
    % that w has the same size as the inputs and each entry of w
    % corresponds to the infoweight for the corresponding triplet (kappa1,
    % kappa2, p).
    w = w(J);
    w = reshape(w, sz);
    
end
