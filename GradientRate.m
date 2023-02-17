function G = GradientRate( P_tx, noise_P, x, h, M, a, weights )
    G       = 0;
    for i = 1:M
        C   = GradientCapacity(P_tx,noise_P,x,h,i,M);
        D   = GradientDispersion(P_tx,noise_P,x,h,i,M);
        G   = G + weights(i)*(C - a(i)*D);
    end
end

function G = GradientCapacity( P_tx, noise_P, x, h, i, M )
    x       = conj(x);
    a       = 0;
    u       = i:M;
    v       = i+1:M;
    for j = v
       a    = a + P_tx*(h(:,j)*h(:,j)'*x);
    end
    G       = P_tx*(h(:,i)*h(:,i)'*x) - P_tx*(x'*h(:,i)*h(:,i)'*x)*a/(noise_P + sum(P_tx*(x'*h(:,v)*h(:,v)'*x)));
    G       = 2*G/(noise_P + sum(P_tx*(x'*h(:,u)*h(:,u)'*x)));
    G       = G/log(2);
end

function G = GradientDispersion( P_tx, noise_P, x, h, i, M )
    x       = conj(x);
    b       = 0;    
    v       = i:M;       
    for j = v
       b    = b + P_tx*(h(:,j)*h(:,j)'*x); 
    end
    g       = noise_P + sum(P_tx*(x'*h(:,v)*h(:,v)'*x));
    G       = P_tx*(h(:,i)*h(:,i)'*x)/g - P_tx*(x'*h(:,i)*h(:,i)'*x)*b/g^2;
    G       = sqrt(2)*G*(P_tx*(x'*h(:,i)*h(:,i)'*x)/g)^(-0.5);
end