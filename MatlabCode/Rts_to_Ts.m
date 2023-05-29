function [Ts] = Rts_to_Ts(Rs, ts)

n = length(Rs);
if (~exist('ts','var'))
    ts = cell(n);
    for i=1:n
        ts{i} = zeros(3,1);
    end
end

Ts = cell(n);

for i=1:n
    T = eye(4);
    R = Rs{i};
    t = ts{i};
    T(1:3,1:3) = R;
    T(1:3,4) = t(:);
    Ts{i} = T;
end

end