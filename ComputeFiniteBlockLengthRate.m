function [ R ] = ComputeFiniteBlockLengthRate( SINR, n, PER )

C = log2(1 + SINR);
V = 2*SINR./(1 + SINR)*log2(exp(1))^2; 
R = C - sqrt(V/n)*qfuncinv(PER) + log2(n)/n;

end

