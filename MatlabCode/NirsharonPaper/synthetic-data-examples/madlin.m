%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function[X,resnorm,residual,exitflag,output,lambda]=madlin(C,d,A,b,Aeq,beq,lb,ub,X0,options)
%MADLIN Constrained linear minimum absolute deviation (L1 norm).
% X=MADLIN(C,d,A,b) solves the minimum absolute deviation problem
%
% min sum(abs(C*x-d)) subject to A*x <= b
% x
%
% where C is m-by-n.
%
% X=MADLIN(C,d,A,b,Aeq,beq) solves the minimum absolute deviation
% (with equality constraints) problem:
%
% min sum(abs(C*x-d)) subject to 
% x A*x <= b and Aeq*x = beq
%
% X=MADLIN(C,d,A,b,Aeq,beq,LB,UB) defines a set of lower and upper
% bounds on the design variables, X, so that the solution 
% is in the range LB <= X <= UB. Use empty matrices for 
% LB and UB if no bounds exist. Set LB(i) = -Inf if X(i) is unbounded 
% below; set UB(i) = Inf if X(i) is unbounded above.
%
% X=MADLIN(C,d,A,b,Aeq,beq,LB,UB,X0) sets the starting point to X0.  This
% option is only available with the active-set algorithm. The default
% interior point algorithm will ignore any non-empty starting point.
%
% X=MADLIN(C,d,A,b,Aeq,Beq,LB,UB,X0,OPTIONS) minimizes with thedefault 
% optimization parameters replaced by values in the structure OPTIONS,an 
% argument created with the OPTIMSET function. See OPTIMSET fordetails. 
% Use options are Display, Diagnostics, TolFun, LargeScale, MaxIter. 
% Currently, only 'final' and 'off' are valid values for the parameter 
% Display when LargeScale is 'off' ('iter' is valid when LargeScale is'on').
%
% [X,RESNORM]=MADLIN(C,d,A,b) returns the value of the 1-norm of the
% residual: sum(abs(C*X-d)).
%
% [X,RESNORM,RESIDUAL]=LSQLIN(C,d,A,b) returns the residual: C*X-d.
%
% [X,RESNORM,RESIDUAL,EXITFLAG] = MADLIN(C,d,A,b) returns EXITFLAG that 
% describes the exit condition of LINPROG.
% If EXITFLAG is:
% > 0 then MADLIN converged with a solution X.
% 0 then MADLIN reached the maximum number of iterations without converging.
% < 0 then the problem was infeasible or MADLIN failed.
%
% [X,RESNORM,RESIDUAL,EXITFLAG,OUTPUT] = MADLIN(C,d,A,b) returns astructure
% OUTPUT with the number of iterations taken in OUTPUT.iterations, thetype
% of algorithm used in OUTPUT.algorithm, the number of conjugategradient
% iterations (if used) in OUTPUT.cgiterations. The minimum absolutedeviation
% problem is posed as a LP; the LP is returned in OUTPUT.lp.f,OUTPUT.lp.A,
% OUTPUT.lp.b, OUTPUT.lp.Aeq, OUTPUT.lp.beq, OUTPUT.lp.lb, and OUTPUT.lp.ub.
%
% [X,RESNORM,RESIDUAL,EXITFLAG,OUTPUT,LAMBDA]=MADLIN(C,d,A,b) returnsthe set of 
% Lagrangian multipliers LAMBDA, at the solution of the LP:LAMBDA.ineqlin 
% for the linear inequalities A, LAMBDA.eqlin for the linearequalities Aeq, 
% LAMBDA.lower for LB, and LAMBDA.upper for UB.
% 
% NOTE: the LargeScale (the default) version of MADLIN uses aprimal-dual
% method. Both the primal problem and the dual problem must befeasible 
% for convergence. Infeasibility messages of either the primalor dual, 
% or both, are given as appropriate. The primal problem instandard 
% form is 
% min f'*x such that A*x = b, x >= 0.
% The dual problem is
% max b'*y such that A'*y + s = f, s >= 0.

% subordinate functions: linprog (Optimization Toolbox)
%
% author: Nathan Cahill
% email: nathan.cahill@kodak.com

% Handle missing arguments
if nargin < 10, options = [];
   if nargin < 9, X0 = []; 
      if nargin < 8, ub=[]; 
         if nargin < 7, lb = []; 
            if nargin < 6, beq =[]; 
               if nargin < 5, Aeq = [];
                  if nargin < 4, b = [];
                     if nargin < 3, A = [];
                     end, end, end, end, end, end, end, end

% Pose problem as LP ("A Global and Quadratic Affine Scaling Method for 
% Linear L1 Problems", Coleman and Li, Cornell University TR 89-1026,
% July 1989).

% The MAD problem min(sum(abs(Cx-d))) can be posed as the following LP:
% min(sum(u+v)) subject to Cx-u+v=b, u>=0, v>=0.
% Any extra equality and inequality constraints can be appended to theLP.

[m,n]=size(C);

% objective function for LP
lp.f=[zeros(n,1);ones(2*m,1)];
lp.b=b;
lp.beq=[beq;d];

% set up constraints sparsely if needed
switch issparse(C)|issparse(A)|issparse(Aeq)
case 0 % set up LP as full
   lp.A=[A zeros(size(A,1),2*m)];
   if isempty(Aeq)
      lp.Aeq=[C -eye(m) eye(m)];
   else
      lp.Aeq=[Aeq zeros(size(Aeq,1),2*m);C -eye(m) eye(m)];
   end
case 1 % set up LP as sparse
   lp.A=[A sparse([],[],[],size(A,1),2*m)];
   if isempty(Aeq)
      lp.Aeq=[C -speye(m) speye(m)];
   else
      lp.Aeq=[Aeq sparse([],[],[],size(Aeq,1),2*m);C -speye(m)
speye(m)]; 
   end
end

% set up bounds for x, u and v
lp.lb=[lb(:);repmat(-inf,n-length(lb),1);zeros(2*m,1)];
lp.ub=ub;

% append u and v initial guesses to X0 if needed
if ~isempty(X0)
   X0=[X0(:);zeros(2*m,1)];
end

% now solve LP
optimset('MaxIter',30);
[X,resnorm,exitflag,output,lambda]=linprog(lp.f,lp.A,lp.b,...
   lp.Aeq,lp.beq,lp.lb,lp.ub,X0,options);

% strip away u and v vectors from solution
X=X(1:n);

% compute residuals
residual=C*X-d;

% append LP problem parameters to output structure
output.lp=lp;

