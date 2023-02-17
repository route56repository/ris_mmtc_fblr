function [ conv_tol, exec_time, iteration ] = ComplexityAnalysisSO( L, M, RIS, h, P_tx, noise_P, n, PER, option, weights )

% Parameters
tol = 1e-4;
mit = 1e3;
mu  = 1e-1;

% Channels
H_i         = cell(M,1);
a           = zeros(1,M);
for i = 1:M
    a(i)    = log2(exp(1))/sqrt(n)*qfuncinv(PER);
    H_i{i}  = P_tx*h(:,i)*h(:,i)'; 
end

H_j         = cell(M,1);
H           = cell(M,1);
for i = 1:M
    aux     = 0;    
    v       = i + 1:M;    
    for j = v
        aux = aux + H_i{j};
    end
    H_j{i}  = aux;
    H{i}    = H_i{i} + H_j{i};
end

% Real values (Shannon and Polyanskiy)
R_Shannon_t         = cell(M,1);
for i = 1:M
    C               = @(X) CapacityReal(H_i{i},H_j{i},X,noise_P);
    R_Shannon_t{i}  = @(X) C(X)/log(2);
end

R_Poly_t    = cell(M,1);
for i = 1:M
    C               = @(X) CapacityReal(H_i{i},H_j{i},X,noise_P);
    D               = @(X) DeltaReal(H_i{i},H{i},X,noise_P);
    R_Poly_t{i}     = @(X) C(X)/log(2) - a(i)*D(X) + log2(n)/n;
end

% Lower bound (Polyanskiy only)
R_Poly_lb           = cell(M,1);
for i = 1:M
    C               = @(X,Xo) CapacityLowerBound(H_i{i},H_j{i},H{i},X,Xo,noise_P);
    D               = @(X,Xo) DeltaUpperBound(H_i{i},H{i},X,Xo,noise_P);
    R_Poly_lb{i}    = @(X,Xo) C(X,Xo) - a(i)*D(X,Xo) + log2(n)/n;
end

%% Solution with SO (+ SDR)
% Functions
if option == 1                                                             % WSR
    R_Poly_lb       = @(X,Xo,weights) FunctionSum(R_Poly_lb,X,Xo,weights);
elseif option == 2                                                         % Minimax
    R_Poly_lb       = @(X,Xo) FunctionMin(R_Poly_lb,X,Xo);
end
    
% Initial Point
Xo          = conj(diag(RIS))*diag(RIS).';
X           = Xo;
R_old       = 0;
if option == 1
    R_new   = R_Poly_lb(Xo,Xo,weights);
elseif option == 2
    R_new   = R_Poly_lb(Xo,Xo);
end

% Iterate (CVX)
counter         = 0;
conv_tol        = R_new;
exec_time       = 0; 
while abs((R_new - R_old)/R_old) > tol && counter < mit
    counter     = counter + 1;
    R_old       = R_new;
    tic
    
    cvx_begin
    cvx_quiet(true)
    cvx_precision best
    cvx_solver sedumi
    variable X(L,L) complex semidefinite
    if option == 1
        R       = 0;
    elseif option == 2
        R       = [];
    end
    for i = 1:M
        C       = CapacityLowerBound(H_i{i},H_j{i},H{i},X,Xo,noise_P);
        A       = real(trace(X*H_i{i}));
        B       = noise_P + real(trace(X*H{i}));
        Do      = DeltaReal(H_i{i},H{i},Xo,noise_P);
        D       = Do + real(trace(Xo*H_i{i}))/Do * inv_pos(B) + ...
                    1/(Do*real(trace(Xo*H_i{i})))*quad_pos_over_lin(A,B); 
        D       = D/2;
        if option == 1
            R   = R + weights(i)*(C - a(i)*D + log2(n)/n);
        elseif option == 2
            R   = [R C - a(i)*D + log2(n)/n];
        end
    end
    if option == 2
        R       = min(R);
    end
    maximize R
    subject to
    diag(X) <= 1;
    diag(X) >= 0;            
    cvx_end
    if ~isequal(cvx_status,'Solved')
        R_new   = R_old;
        X       = Xo;
        counter = -counter;
        continue;
    end
    
    % Check constraints
    if any(eig(X) < 0 - tol) || any(diag(X) < 0 - tol | diag(X) > 1 +  tol)
        error('Constraints unsatisfied');
    end
    
    % Update variables
    if option == 1
        R_new   = R_Poly_lb(X,Xo,weights);
    elseif option == 2
        R_new   = R_Poly_lb(X,Xo);
    end

    % Check for numerical problems
    if abs(R_new - R)/R > mu      
        R_new   = R_old;
        X       = Xo;
        counter = -counter;
        continue;
    end

    Xo          = X;
    if option == 1
        R_new   = R_Poly_lb(X,Xo,weights);
    elseif option == 2
        R_new   = R_Poly_lb(X,Xo);
    end
    
    conv_tol    = [conv_tol R_new];
    exec_time   = [exec_time toc];
end

% Save results
conv_tol    = (conv_tol(end) - conv_tol)/conv_tol(end);
exec_time   = cumsum(exec_time);
iteration   = counter;

end

function y = FunctionSum( varargin )
   y            = 0;
   if nargin == 3
        f       = varargin{1};
        X       = varargin{2};
        w       = varargin{3};
        for i = 1:numel(f)
            y   = y + w(i)*f{i}(X);
        end
   elseif nargin == 4
        f       = varargin{1};
        X       = varargin{2};
        Xo      = varargin{3};
        w       = varargin{4};
        for i = 1:numel(f)
            y   = y + w(i)*f{i}(X,Xo);
        end
   end
end

function [y,i] = FunctionMin( varargin )
   if nargin == 2
        f           = varargin{1};
        X           = varargin{2};
        y           = zeros(1,numel(f));
        for i = 1:numel(f)
            y(i)    = f{i}(X);
        end
        [y,i]       = min(y);
   elseif nargin == 3
        f           = varargin{1};
        X           = varargin{2};
        Xo          = varargin{3};
        y           = zeros(1,numel(f));
        for i = 1:numel(f)
            y(i)    = f{i}(X,Xo);
        end
        [y,i]       = min(y);
   end
end

% Real magnitudes
function C = CapacityReal( H_i, H_j, X, noise_P )
    C = log(1 + real(trace(X*H_i))/(noise_P + real(trace(X*H_j)))); 
end

function D = DeltaReal( H_i, H, X, noise_P )
    D = sqrt(2*real(trace(X*H_i))/(noise_P + real(trace(X*H))));
end

% Bounds
function C = CapacityLowerBound( H_i, H_j, H, X, Xo, noise_P )
    Co  = CapacityReal(H_i,H_j,Xo,noise_P);
    C   = Co + real(trace(Xo*H_i))/(noise_P + real(trace(Xo*H_j)))*...
            (2*sqrt(real(trace(X*H_i)))/sqrt(real(trace(Xo*H_i))) - ...
            (noise_P + real(trace(X*H)))/(noise_P + real(trace(Xo*H))) - 1);
    C   = C/log(2);
end

function D = DeltaUpperBound( H_i, H, X, Xo, noise_P )
    Do  = DeltaReal(H_i,H,Xo,noise_P);
    D   = Do + real(trace(Xo*H_i))/Do * 1/(noise_P + real(trace(X*H))) + ...
            1/(Do*real(trace(Xo*H_i)))*real(trace(X*H_i))^2/(noise_P + real(trace(X*H)));
    D   = D/2;
end