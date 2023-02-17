function [ s ] = ComputeSteeringVectorUPA( D, lambda, L, el_ang, aoa )

s = exp(1i*2*pi*D/lambda*((0:L-1)*cos(el_ang)*sin(aoa) + (0:L - 1)*sin(el_ang))).';

end