function [G, W, y, out, b, theta] = syncL1(NN,d, G, R, Weight, opts)
% 
% Given a graph G=({1,2,...,n}, E)
%     
% R_{ij} =  0 if (ij) is not in E.
%     = relative ratio if (ij) is in E
% R_{ii} = I_d
%     
% where R is a Nd x Nd big matrix and N is the number of the rotations.
%------------------------------------------------------------------
% ADMM for synchronization over O(d) 
%
% min  sum_{i,j \in E} Weight_ij ||R_ij - G_ij||
% s.t. A(G) = b, G psd
%
% Weight_ii=1, Weight_ij=0 if i,j not in E
%
% Author: Lanhui Wang
% date: 7/30, 2012
%------------------------------------------------------------------
% See details in syncL1.m
%------------------------------------------------------------------
% set up parameters
if ~isfield(opts, 'tol');    opts.tol = 1e-3;   end
if ~isfield(opts, 'mu');     opts.mu = 1;       end
if ~isfield(opts, 'gam');    opts.gam = 1.618;  end
if ~isfield(opts, 'EPS');    opts.EPS = 1e-12;  end
if ~isfield(opts, 'maxit');  opts.maxit = 1000; end
if ~isfield(opts, 'record'); opts.record = 0;   end
if ~isfield(opts, 'warning'); opts.warning = 1;  end 
if ~isfield(opts, 'sparse_structure'); opts.sparse_structure = 0;  end

if opts.warning==0
    warning off;
end

if ~isfield(opts, 'adp_proj');  opts.adp_proj = 1; end %1 or 0
if ~isfield(opts, 'max_rankW'); opts.max_rankW = max(d*2, floor(NN/2)); end

tol     = opts.tol;
mu      = opts.mu;
gam     = opts.gam;
EPS     = opts.EPS;
maxit   = opts.maxit;
record  = 0; %opts.record;   % Nir

adp_proj = opts.adp_proj;
max_rankW = opts.max_rankW;


% parameters for adjusting mu
if ~isfield(opts, 'adp_mu');        opts.adp_mu = 1;        end %1 or 0
if ~isfield(opts, 'dec_mu');        opts.dec_mu = 0.5;      end
if ~isfield(opts, 'inc_mu');        opts.inc_mu = 2;        end
if ~isfield(opts, 'mu_min');        opts.mu_min = 1e-4;     end
if ~isfield(opts, 'mu_max');        opts.mu_max = 1e4;      end
if ~isfield(opts, 'min_mu_itr');    opts.min_mu_itr = 5;    end
if ~isfield(opts, 'max_mu_itr');    opts.max_mu_itr = 20;   end
if ~isfield(opts, 'delta_mu_l');    opts.delta_mu_l = 0.1;  end
if ~isfield(opts, 'delta_mu_u');    opts.delta_mu_u = 10;   end
if ~isfield(opts, 'max_nev');       opts.max_nev = 6;   end

adp_mu  = opts.adp_mu;
dec_mu  = opts.dec_mu;  
inc_mu  = opts.inc_mu;  
mu_min  = opts.mu_min;   
mu_max  = opts.mu_max;
max_mu_itr = opts.max_mu_itr;
delta_mu_l = opts.delta_mu_l;
delta_mu_u = opts.delta_mu_u;
max_nev = opts.max_nev;
sparse_structure = opts.sparse_structure;

itmu_pinf = 0;  
itmu_dinf = 0;

% Use the sparse structure
if sparse_structure == 1
    R = sparse (R);
    [idr,idc]=find(R);
end
%------------------------------------------------------------------
% set up the SDP problem
n = d*NN;

% indices for entries in R
if sparse_structure ==1
    ind_nnz = sub2ind([n n],idr,idc);
    ind_z = setdiff(1:n^2, ind_nnz);
else
    ind_z = (R==0);
    ind_nnz = (R~=0);
end
% set up linear constraints
idx = 1:n; 
[row1,col1]=find(triu(ones(d),1));
l=length(row1);
row1 = repmat(row1,NN,1);
col1 = repmat(col1,NN,1);
for di = 2:NN
    row1([1:l]+(di-1)*l) = row1([1:l]+(di-1)*l)+(di-1)*d;
    col1([1:l]+(di-1)*l) = col1([1:l]+(di-1)*l)+(di-1)*d;
end
col = (col1-1)*n+row1;

m = n + l*NN;
b = [ones(n,1); zeros(l*NN,1)];
% intial values
W = eye(n);

% provide an initial theta here:
Phi = W+G/mu;
if sparse_structure == 1
    Phi = sparse(idr,idc,Phi(ind_nnz));
else
    Phi(ind_z) = 0;
end
theta=R2theta(Phi,R,mu,d,ind_nnz);
% then compute S = Q(theta)
Weight=kron(Weight,ones(d));
S = Qtheta(theta, Weight);
AS = ComputeAX(S);

resi = ComputeAX(G) - b;

kk = 0; nev = 0;
opteigs.maxit = 100; opteigs.issym = 1;
opteigs.isreal = 1; opteigs.disp = 0; opteigs.tol = 1e-6;

%------------------------------------------------------------------
if record >= 1
    fprintf('%4s %8s %10s %10s %10s %10s %10s\n', ...
        'itr', 'mu', 'pobj', 'dobj', 'gap', 'pinf', 'dinf');
end

% main routine
for itr = 1:maxit
    %------------------------------------------------------------------
    % compute y
    y = -(AS + ComputeAX(W)) - resi/mu;
    
    %******************************
    % compute theta
    ATy = ComputeATy(y);
    Phi = W + ATy + G/mu;
    if sparse_structure == 1
        Phi = sparse(idr,idc,Phi(ind_nnz));
    else
        Phi(ind_z) = 0;
    end
    theta = R2theta(Phi,R,mu,d,ind_nnz);
    % then compute S = Q(theta)
    S=Qtheta(theta, Weight);
    %******************************
    
    % compute W
    H = - S - ATy - G/mu;     H = (H + H')/2;
    if adp_proj == 0
        [V,D] = eig(H);
        D = diag(D);
        nD = D>EPS;
        nev = nnz(nD);
        if nev < n/2 % few positive eigenvalues
            W = V(:,nD)*diag(D(nD))*V(:,nD)';
            WmH = W - H;
        else  % few negative eigenvalues
            nD = ~nD;
            WmH = V(:,nD)*diag(-D(nD))*V(:,nD)';
            W = WmH + H;
        end
    elseif adp_proj == 1 
        % compute G = W - H since G is expected to be rank d
        % estimate rank, d*2 is a safeguard here
        if itr == 1
            nev = max_rankW;
        else
            if  nev > 0
                %nev
                drops = dH(1:end-1)./dH(2:end);
                [dmx,imx] = max(drops);  
                rel_drp = (nev-1)*dmx/(sum(drops)-dmx);
                %rel_drp
                if rel_drp > 50
                    nev = max(imx, max_nev); 
                else
                    nev = nev + d*2-1;
                end
                %nev
            else
                nev = d*2;
            end
        end
        nev = min(nev, n);
        % computation of W and H
        [V,dH] = eigs(-H, nev,'la', opteigs);
        dH = diag(dH); nD = dH>EPS; nev = nnz(nD); 
        %dH'
        if nev > 0
            dH = dH(nD);
            WmH = V(:,nD)*diag(dH)*V(:,nD)';
            W = WmH + H;
        else
            WmH = sparse(n,n);
            W = H;
        end
    end
    
    % update G
    %G = (1-gam)*G + gam*mu*(W-H);
    G = (1-gam)*G + gam*mu*WmH;
    
    %------------------------------------------------------------------
    % check optimality
    if sparse_structure == 1
        temp = R - sparse(idr,idc,G(ind_nnz));
    else
        temp = R-G;
        temp(ind_z)=0;
    end
    temp = Weight.*temp;
    temp = temp.^2;
    temp2 = temp(1:d:end,1:d:end);
    for j1=1:d
        for j2 = 1:d
            if j1>1 || j2>1
                temp2 = temp2 + temp(j1:d:end,j2:d:end);
            end
        end
    end
    pobj=sum(sqrt(temp2(:)))/2;
    dobj = -full(b'*y)-sum(sum(Weight.*theta.*R))/2;
    gap  = abs(dobj-pobj)/(1+abs(dobj)+abs(pobj));
   
    resi = ComputeAX(G) - b;
    nrmb = max(norm(b),1);
    nrmS = max(norm(S,inf),1);
    
    pinf = norm(resi)/nrmb;
    dinf = norm(S+W+ATy,'fro')/nrmS;
    
%    ainf = max([pinf,dinf,gap]);
    ainf = max([pinf,dinf]);
    dtmp = pinf/dinf;

    if record >= 1
        %fprintf('%4d   %3.2e   %+3.2e   %+3.2e   %+3.2e   %3.2e   %+3.2e   %+3.2e  %d  %d  %d %d\n', ...
        %        itr, mu,  pobj,  dobj, gap, pinf, dinf,  dtmp, kk, nev, itmu_pinf, itmu_dinf);
    
        fprintf('%4d   %3.2e   %3.2e   %+3.2e   %+3.2e  %d  %d  %d %d\n', ...
                itr, mu,  pinf, dinf,  dtmp, kk, nev, itmu_pinf, itmu_dinf);
    
    end

    if ainf <= tol
        out.exit = 'optimal';
        break;
    end
    

    % update mu adpatively
    if adp_mu == 1
        if (dtmp <= delta_mu_l)
            itmu_pinf = itmu_pinf + 1;  itmu_dinf = 0;
            if itmu_pinf > max_mu_itr
                %mu = max(mu*dec_mu, mu_min); itmu_pinf = 0;
                mu = max(mu*inc_mu, mu_min); itmu_pinf = 0;
            end
        elseif dtmp > delta_mu_u
            itmu_dinf = itmu_dinf + 1;  itmu_pinf = 0;
            if itmu_dinf > max_mu_itr
                %mu = min(mu*inc_mu, mu_max); itmu_dinf = 0;
                mu = min(mu*dec_mu, mu_max); itmu_dinf = 0;
            end
        end
    end
  
end

if opts.warning==0
    warning on;
end

out.pobj = pobj;
out.dobj = dobj;
out.gap  = gap;
out.itr = itr;
out.pinf = pinf;
out.dinf = dinf;
% End main routine
%--------------------------------------------------------------------------


	% compute A'*y 
    function ATy = ComputeATy(y)
        ATy = sparse(n,n);
        ATy(col) = (sqrt(2)/2)*y(n+1:m);
        ATy = ATy + ATy'+ sparse(idx, idx, y(1:n), n, n);
    end

    % compute A*X
    function AX = ComputeAX(X)
        AX = [spdiags(X,0); sqrt(2)*X(col)];
    end

    function theta = R2theta(Phi, R, mu,d, ind_nnz)
        % Update theta
        %
        % Lanhui Wang
        % Jul 27, 2012
%         K = size(R,1)/d;
        %         theta = zeros(3*K,3*K);
        theta = R - mu* Phi;
        temp = theta;
        temp = temp.*temp;
        temp2 = temp(1:d:end,1:d:end);
        for m1=1:d
            for m2 = 1:d
                if m1>1 || m2>1
                temp2 = temp2 + temp(m1:d:end,m2:d:end);
                end
            end
        end
        temp = sqrt(temp2);
        temp = kron(temp,ones(d));
        theta(ind_nnz) = theta(ind_nnz) ./temp(ind_nnz);
        
    end


    function S = Qtheta(theta,Weight)
        % Update S
        %
        % Lanhui Wang
        % Jul 19, 2012
        
        S = Weight.*theta/2;
    end



end