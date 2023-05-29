function out = logarithm_map(in)
    cos_theta = (trace(in)-1)/2;
    sin_theta = sqrt(1-cos_theta^2);
    theta = acos(cos_theta);
    ln_R = theta/(2*sin_theta)*(in-in');
    out = [ln_R(3,2);ln_R(1,3);ln_R(2,1)];
end
