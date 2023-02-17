function [ conv_tol, exec_time, iteration ] = ComplexityAnalysisAO( L, M, RIS, h, P_tx, noise_P, n, PER, option, weights )

% Parameters
tol = 1e-4;
mit = 1e3;
N   = 1e2;

% Phase-shifts
ang         = linspace(0,2*pi,N + 1);
ang(end)    = [];

% Amplitude coefficients
lambda      = linspace(0,1,N);
values      = repmat(lambda,N,1).*repmat(exp(1i*ang),N,1).';
values      = unique(values(:));
K           = length(values);

if any(abs(values) < 0 - tol | abs(values) > 1 + tol)
    error('Search space unfeasible');
end

%% Solution with Alternating Optimization
% Initial point
R_old       = 0;
SINR        = ComputeSINR(M,1,h,RIS,P_tx,noise_P);
if option == 1                                                             % WSR
    R_new   = sum(weights.*ComputeFiniteBlockLengthRate(SINR,n,PER));
elseif option == 2                                                         % minimax
    R_new   = min(ComputeFiniteBlockLengthRate(SINR,n,PER));
end
counter     = 0;
conv_tol    = R_new;
exec_time   = 0;

% Iterate
while abs((R_new - R_old)/R_old) > tol && counter < mit
    counter = counter + 1;
    R_old   = R_new;
    tic
    
    for i = 1:L
        c_R = zeros(1,K);
        for k = 1:K
            RIS(i,i)    = values(k);
            SINR        = ComputeSINR(M,1,h,RIS,P_tx,noise_P);
            if option == 1
                c_R(k)  = sum(weights.*ComputeFiniteBlockLengthRate(SINR,n,PER));
            elseif option == 2
                c_R(k)  = min(ComputeFiniteBlockLengthRate(SINR,n,PER));
            end
        end
        % Update variables
        [~,ind]         = max(c_R);
        RIS(i,i)        = values(ind);
    end
    SINR                = ComputeSINR(M,1,h,RIS,P_tx,noise_P);
    if option == 1
        R_new           = sum(weights.*ComputeFiniteBlockLengthRate(SINR,n,PER));
    elseif option == 2
        R_new           = min(ComputeFiniteBlockLengthRate(SINR,n,PER));
    end
    
    conv_tol            = [conv_tol R_new];
    exec_time           = [exec_time toc];
end

% Save results
conv_tol    = (conv_tol(end) - conv_tol)/conv_tol(end);
exec_time   = cumsum(exec_time);
iteration   = counter;

end