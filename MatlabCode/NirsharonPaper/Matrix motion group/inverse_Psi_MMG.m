function [ MMG_element ] = inverse_Psi_MMG(Q, d, lambda)
% inverse mapping for MMG

%[mu1, mu2, B] = inverse_Cartan_MMG(Q, d);
[mu1, mu2, B] = inverse_Cartan_MMG_imp(Q, d);
if cost_funV1(B,Q,d)>1e-30
    [mu1, mu2, B] = inverse_Cartan_MMG_imp(Q, d);
end
MMG_element = {mu1, mu2, B*lambda};

end

