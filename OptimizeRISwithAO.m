function [ rate, RIS ] = OptimizeRISwithAO( L, M, RIS, h, P_tx, noise_P, n, PER, option, weights )

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
elseif option == 2                                                         % Minimax
    R_new   = min(ComputeFiniteBlockLengthRate(SINR,n,PER));
end

% Iterate
counter     = 0;
while abs((R_new - R_old)/R_old) > tol && counter < mit
    counter = counter + 1;
    R_old   = R_new;
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
end

% Save results
SINR                    = ComputeSINR(M,1,h,RIS,P_tx,noise_P);
rate(1)                 = sum(weights.*ComputeFiniteBlockLengthRate(SINR,n,PER));
rate(2)                 = min(ComputeFiniteBlockLengthRate(SINR,n,PER));

end