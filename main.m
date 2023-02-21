%% --------------------------------------------------------------------- %%
%            Rate Optimization for RIS-Aided mMTC Networks in             %
%                      the Finite Blocklength Regime                      %
%% --------------------------------------------------------------------- %%

clear; close all;
delete(gcp('nocreate'));

cvx_setup;
parpool;

rng(1);

%% Simulation Options
opt_sr  = input('Select weight criteria, equality (1) or fairness (2): '); % Weighted sum rate option: Equal (1), or fair (2) weights

%% System Parameters
% Variable
M       = 1e1;                                                             % Number of sensors
n       = 1e2;                                                             % Packet length
L_t     = [1 2.5*(10:10:100)];                                             % Number of RIS elements
r       = 1e1;                                                             % Deployment radius
std_SR  = 1;                                                               % Std of sensor-to-RIS channel                        
std_RC  = 1;                                                               % Std of RIS-to-CN channel
K       = 1e3;                                                             % Number of realizations
PER     = 1e-3;                                                            % PER in the FBLR
min_dst = 1e0;                                                             % Minimum distance to RIS and CN
beta    = 0;                                                               % Percentage of I-CSI errors

% Fixed
F       = [10 1];                                                          % Rician factors (RIS-CN and sensors-RIS)
BW      = 1.08e6;                                                          % System bandwidth
P_tx    = 10^(0/10)*1e-3;                                                  % Transmit power
N_o     = 10^(-174/10)*1e-3;                                               % Thermal noise spectral density [dBm/Hz]
CN_NF   = 10^(5/10);                                                       % CN noise figure
shadow  = 10^(8/10);                                                       % Lognormal shadowing std 
alpha   = 3;                                                               % Decay factor
f_o     = 1.9e9;                                                           % Carrier frequency
c_o     = 3e8;                                                             % Light speed
lambda  = c_o/f_o;                                                         % Wavelength
D       = lambda/2;                                                        % Distance between array elements
el_ang  = 0;                                                               % Elevation angle
N_o     = N_o*CN_NF*shadow;                                                % Incorporate all effects into N_o
G_CN    = 2.15;                                                            % CN antenna gain [dB]
G_SNS   = 2.15;                                                            % Sensors antenna gain [dB]
G_RIS   = 17;                                                              % RIS antenna gain [dB]

% Auxiliar
AG_SNS  = 10^((G_SNS + G_CN)*0.1);                                         % Antenna Gain in sensors to CN (BS) link
AG_RIS  = 10^((G_SNS + G_RIS)*0.1);                                        % Antenna Gain in sensors to RIS link
AG_CN   = 10^((G_RIS + G_CN)*0.1);                                         % Antenna Gain in RIS to CN link
noise_P = N_o*BW;                                                          % Noise power (AWGN)
  
%% Magnitudes
% 1. Normalize by noise_P to avoid numerical problems/issues
P_tx    = P_tx/noise_P;
P_n     = noise_P;
noise_P = 1;

% 2. Rates
N_optx  = 5;                                                               % AO, EGA, RGA, SO and Shannon SO
N_other = 8;                                                               % Optimized with rate or shannon: FBLR, capacity + w & w/o rank-one constraint
Rate_o  = cell(length(L_t),1);  
Rate    = cell(length(L_t),N_optx);  
Other   = cell(length(L_t),N_other);  

% 3. Complexity
Conv_Tol    = cell(N_optx,1);  
Exec_Time   = cell(N_optx,1);
Iter        = cell(N_optx,1);

% 4. Positions
pos_RIS                 = [r r];
pos_sns                 = zeros(M,2,K);
for k = 1:K
    Z                   = zeros(M,2);
    d_R                 = zeros(M,1);
    while any(d_R < min_dst)        
        a               = rand(M,1) * 2 * pi;
        R               = r * sqrt(rand(M,1));
        X               = R .* cos(a) + pos_RIS(1);
        Y               = R .* sin(a) + pos_RIS(2);        
        Z               = [X Y]; 
        for i = 1:M
            d_R(i)      = norm(Z(i,:) - pos_RIS);
        end
        pos_sns(:,:,k)  = Z;
    end
end

% 5. Distances and angles
% Between RIS and CN
dist_RC = norm(pos_RIS);                                               
ang_RC  = atan2(pos_RIS(2),pos_RIS(1));                                

% Between sensors and RIS
dist_SR             = zeros(M,K);                                           
ang_SR              = zeros(M,K);
for k = 1:K
    for i = 1:M
        dist_SR(i,k) = norm(pos_sns(i,:,k) - pos_RIS);
        ang_SR(i,k)  = atan2(pos_sns(i,2,k) - pos_RIS(2),pos_sns(i,1,k) - pos_RIS(1));
    end
end

% Normalized distances
dist_RC = dist_RC/((lambda/(4*pi)));                                       
dist_SR = dist_SR/((lambda/(4*pi)));                                       

%% Sweep (over L)
disp('------------------------------------------------------------------');
disp(['Number of realizations: ' num2str(K)]);
disp('------------------------------------------------------------------');    

disp('------------------------------------------------------------------');
disp(['Number of TX symbols: ' num2str(n)]);
disp('------------------------------------------------------------------');    

disp('------------------------------------------------------------------');
disp(['Number of sensors: ' num2str(M)]);
disp('------------------------------------------------------------------');    

disp('------------------------------------------------------------------');
disp(['Transmit power: ' num2str(10*log10(P_n*P_tx*1e3)) ' dBm']);
disp('------------------------------------------------------------------');   

c_L = 0;
for L = L_t
c_L = c_L + 1;
disp('------------------------------------------------------------------');
disp(['Number of RIS elements: ' num2str(L)]);
disp('------------------------------------------------------------------');    

% 6. Channels
N_paths = L;
kappa_t = 6;

% RIS-CN link
g_RC                = zeros(L,K);
phi_RC              = ang_RC;
path_RC             = sqrt(AG_CN)*dist_RC^(-alpha/2);                      % Path-loss
for k = 1:K
    phi             = circ_vmrnd(phi_RC,kappa_t,N_paths);                  % AoAs
    for np = 1:N_paths                                                     % Steering vector
        v_RC        = ComputeSteeringVectorUPA(D,lambda,L,el_ang,phi(np));
        g_RC(:,k)   = g_RC(:,k) + v_RC;
    end
    g_RC(:,k)       = g_RC(:,k)/sqrt(N_paths);
    aux             = 1/sqrt(2) * (randn(1) + 1i*randn(1)); 
    aux             = std_RC*aux;                                        % Fading
    g_RC(:,k)       = sqrt(1/(1 + F(1)))*aux + sqrt(F(1)/(1 + F(1)))*g_RC(:,k);
    g_RC(:,k)       = path_RC*g_RC(:,k);                                          
end

% Sensors to RIS
g_SR                    = zeros(L,M,K);
path_g                  = sqrt(AG_RIS)*dist_SR.^(-alpha/2);                % Path-loss
for k = 1:K
    for i = 1:M
        phi_G           = ang_SR(i,k);
        phi             = circ_vmrnd(phi_G,kappa_t,N_paths);               % AoAs
        for np = 1:N_paths                                                 % Steering vector
            v           = ComputeSteeringVectorUPA(D,lambda,L,el_ang,phi(np));
            g_SR(:,i,k) = g_SR(:,i,k) + v;
        end
        g_SR(:,i,k)     = g_SR(:,i,k)/sqrt(N_paths);
        aux             = 1/sqrt(2) * (randn(1) + 1i*randn(1));
        aux             = std_SR*aux;                                    % Fading (Rice)
        g_SR(:,i,k)     = sqrt(1/(1 + F(2)))*aux + sqrt(F(2)/(1 + F(2)))*g_SR(:,i,k);
        g_SR(:,i,k)     = path_g(i,k)*g_SR(:,i,k);             
    end
end

% Cascaded
h                   = zeros(L,M,K);
for k = 1:K
    for i = 1:M
        h(:,i,k)    = diag(g_RC(:,k))*g_SR(:,i,k);
    end
end

% Covariance matrix
C           = cell(M,1);
for i = 1:M
    aux     = reshape(h(:,i,:),L,K);
    C{i}    = cov(aux.');
end
h_org       = h;

% Add CSI errors
if beta > 0
    for k = 1:K
        for i = 1:M
            h(:,i,k) = h(:,i,k) + sqrt(beta*trace(C{i})/L)/sqrt(2)*(randn(L,1) + 1i*randn(L,1));
        end
    end
end

% 7. Initial RIS
RIS                 = cell(K,N_optx,2);
for k = 1:K
    for j = 1:N_optx
        RIS{k,j,1}  = eye(L);
        RIS{k,j,2}  = eye(L);
    end
end

%% Optimization
rate        = zeros(K,N_optx,2,2);
other       = zeros(K,N_other,2,2);

conv_tol    = cell(K,N_optx - 1);
exec_time   = cell(K,N_optx - 1);
iter        = zeros(K,N_optx - 1);

weights     = zeros(M,K);
rate_o      = zeros(K,2);

% 1. Sensor ordering and weights
for k = 1:K
    order               = 1:M;
    h(:,:,k)            = h(:,order,k); 
    h_org(:,:,k)        = h_org(:,order,k); 
    
    gains               = P_tx*abs(ones(L,1).'*h_org(:,:,k)).^2;
    SINR                = ComputeSINR(M,1,h_org(:,:,k),eye(L),P_tx,noise_P);
    rates               = ComputeFiniteBlockLengthRate(SINR,n,PER);
    if opt_sr == 1                                                         % Equal
        weights(:,k)    = ones(M,1);
    elseif opt_sr == 2                                                     % Fair: inverted SINR
        weights(:,k)    = 1./SINR;
    end
    weights(:,k)        = M*weights(:,k)/sum(weights(:,k));
    rate_o(k,:)         = [sum(weights(:,k).*rates) min(rates)];
end

% 2. Monte-Carlo simulations
parfor k = 1:K
    % WSR and minimax
    for o = 1:2    
        [r1,AO]         = OptimizeRISwithAO(L,M,RIS{k,1,o},h_org(:,:,k),P_tx,noise_P,n,PER,o,weights(:,k));
        if beta > 0
            SINR        = ComputeSINR(M,1,h(:,:,k),AO,P_tx,noise_P);
            r1(1)       = sum(weights(:,k).*ComputeFiniteBlockLengthRate(SINR,n,PER));
            r1(2)       = min(ComputeFiniteBlockLengthRate(SINR,n,PER));
        end
        
        if o == 1
            [r2,~]      = OptimizeRISwithEGA(M,RIS{k,2,o},h_org(:,:,k),P_tx,noise_P,n,PER,weights(:,k));
            [r3,~]      = OptimizeRISwithRGA(M,RIS{k,3,o},h_org(:,:,k),P_tx,noise_P,n,PER,weights(:,k));
        elseif o == 2
            r2          = NaN*ones(size(r1));
            r3          = NaN*ones(size(r1));
        end
    
        [rates_SO,SO]   = OptimizeRISwithSO(L,M,RIS{k,4,o},h_org(:,:,k),P_tx,noise_P,n,PER,o,weights(:,k)); 
        rates_SO(3:4,:) = rates_SO(3:4,:) + [sum(weights(:,k)*log2(n)/n) log2(n)/n];
        r4              = rates_SO(1,:);
        if beta > 0
            SINR        = ComputeSINR(M,1,h(:,:,k),SO,P_tx,noise_P);
            r4(1)       = sum(weights(:,k).*ComputeFiniteBlockLengthRate(SINR,n,PER));
            r4(2)       = min(ComputeFiniteBlockLengthRate(SINR,n,PER));
        end
        
        rates_SSO       = NaN*ones(size(rates_SO));
        r5              = NaN*ones(size(r1));
        
        rate(k,:,o,:)   = [r1',r2',r3',r4',r5']';
        other(k,:,o,:)  = [rates_SO' rates_SSO']';
    end

    % Complexity
    [c1,e1,i1]      = ComplexityAnalysisAO(L,M,RIS{k,1},h(:,:,k),P_tx,noise_P,n,PER,1,weights(:,k));
    [c2,e2,i2]      = ComplexityAnalysisEGA(M,RIS{k,2},h(:,:,k),P_tx,noise_P,n,PER,weights(:,k));
    [c3,e3,i3]      = ComplexityAnalysisRGA(M,RIS{k,3},h(:,:,k),P_tx,noise_P,n,PER,weights(:,k));
    [c4,e4,i4]      = ComplexityAnalysisSO(L,M,RIS{k,4},h(:,:,k),P_tx,noise_P,n,PER,1,weights(:,k)); 
    
    conv_tol(k,:)   = {c1,c2,c3,c4};
    exec_time(k,:)  = {e1,e2,e3,e4}; 
    iter(k,:)       = [i1,i2,i3,i4];
end

% Save results
Rate_o{c_L}         = rate_o;
for j = 1:N_optx
    Rate{c_L,j}     = mean(rate(:,j,:,:),1);
end
for j = 1:N_other
    Other{c_L,j}    = mean(other(:,j,:,:),1);
end

for j = 1:N_optx - 1
    Conv_Tol{j}     = conv_tol(:,j);
    Exec_Time{j}    = exec_time(:,j);
    Iter{j}         = iter(:,j);
end
end