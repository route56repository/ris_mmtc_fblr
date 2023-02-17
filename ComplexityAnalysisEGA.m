function [ conv_tol, exec_time, iteration ] = ComplexityAnalysisEGA( M, RIS, h, P_tx, noise_P, n, PER, weights )

% Parameters
tol = 1e-4;
mit = 1e3;
a_o = 1;
c   = 0.5;
tau = 0.5;

% Auxiliar
a           = zeros(1,M);
for i = 1:M
    a(i)    = log2(exp(1))/sqrt(n)*qfuncinv(PER);
end

%% Solution with Projected Gradient Descend
% Functions and gradients
Grad    = @(x) -conj(GradientRate(P_tx,noise_P,x,h,M,a,weights)); 
Dir     = @(x) -Grad(x);
SINR    = @(x) ComputeSINR(M,1,h,diag(x),P_tx,noise_P);
Fun     = @(x) -sum(weights.*ComputeFiniteBlockLengthRate(SINR(x),n,PER));

% Initial point
R_old   = 0;
alpha   = a_o;
xo      = diag(RIS);
R_new   = -Fun(xo);

% Iterate
counter         = 0;
conv_tol        = R_new;
exec_time       = 0;
while abs((R_new - R_old)/R_old) > tol && counter < mit
    counter     = counter + 1;
    R_old       = R_new;
    tic

    % Search
    alpha       = ArmijoStep(alpha,tau,c,Grad,xo,Dir(xo),Fun,mit);         % Armijo step size
    x           = xo + alpha*Dir(xo);                                      % Gradient descent/ascent
    x           = x/sqrt(max(abs(x).^2));                                  % Retraction

    % Update variables
    R_new       = -Fun(x);
    xo          = x;
    alpha       = a_o;
    
    conv_tol    = [conv_tol R_new];
    exec_time   = [exec_time toc];
end

% Save results
conv_tol    = (conv_tol(end) - conv_tol)/conv_tol(end);
exec_time   = cumsum(exec_time);
iteration   = counter;

end