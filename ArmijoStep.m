function alpha = ArmijoStep( a, tau, c, g, x, p, f, mit )
    m           = g(x).'*p;
    t           = -c*m;
    i           = 0;
    alpha       = tau^(i)*a;
    y           = x + alpha*p;
    y           = y/max(abs(y));
    while f(x) - f(y) < alpha*t && i < mit
        i       = i + 1;
        alpha   = tau^(i)*a;
        y       = x + alpha*p;
        y       = y/max(abs(y));
        if alpha == 0
           continue; 
        end
    end
end