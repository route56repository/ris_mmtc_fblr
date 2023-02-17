function [ SINR ] = ComputeSINR( M, K, h, RIS, P_tx, noise_P )

% Gains
gains           = zeros(M,K);
for k = 1:K
    gains(:,k)  = P_tx*abs(diag(RIS).'*h(:,:,k)).^2;
end

% SINRs
SINR            = zeros(M,K);
aux             = sum(gains);
for i = 1:M
    if i == M
        aux     = 0;
    else
        aux     = aux - gains(i,:);
    end
    SINR(i,:)   = gains(i,:)./(noise_P + aux);
end

end

