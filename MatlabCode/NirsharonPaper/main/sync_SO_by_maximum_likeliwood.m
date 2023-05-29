function [ estimation ] = sync_SO_by_maximum_likeliwood(Affin_mat, confidence_weights, k)
% Wrapping the implementation by Nicolas Boumal of SO sync based on maximum likeliwood
% Works only for cases of d=3,4
%
%
% N.S. April 2016

n = size(Affin_mat,1)/k;

% just a guess, non-outliers
p = 0.75; %p;
% full graph data
[I J] = find(triu(ones(n), 1));
M = n*(n-1)/2;
H = zeros(k,k,M);
% convert the data
counter = 1;
for l=1:n
    for m=(l+1):n
        I(counter) = l;
        J(counter) = m;
        ind1 = 1+(l-1)*k;
        ind2 = 1+(m-1)*k;
        if confidence_weights(l,m)~=0
            H(:,:,counter) = Affin_mat(ind1:(ind1+k-1),ind2:(ind2+k-1));
            counter = counter+1;
        end
    end
end
% define the problem

synchroproblem = buildproblem(k, n, M, I, J, H, .1*ones(M,1), zeros(M,1), p*ones(M, 1) );

%[Rmleplus info params details] = ...
%                           synchronizeMLEplus(synchroproblem, R0, options);

R0   = initialguess(synchroproblem);

options.verbosity = 0;  % other options?
%Rmle = synchronizeMLE(synchroproblem, R0, options);
if k==4
    estimation = synchronizeMLE(synchroproblem, R0, options);
else if k<4
        [Rmleplus, ~, ~, ~] = synchronizeMLEplus(synchroproblem, R0, options);
        estimation = Rmleplus;
    else
        warning('Invalid dimension parameter');
    end
end
end

