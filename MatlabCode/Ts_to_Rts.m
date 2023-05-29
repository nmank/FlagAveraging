function [Rs, ts] = Ts_to_Rts(Ts)

n = length(Ts);
Rs = cell(n);
ts = cell(n);

for i=1:n
    T = Ts{i};
    Rs{i} = T(1:3,1:3);
    ts{i} = T(1:3,4);
end

end