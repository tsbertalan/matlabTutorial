function dTdt = coupledOscRHS(thetas, omegas, K)
    N = numel(thetas);

    dTdt = zeros(size(thetas));
    for i=1:N
        dTdt(i) = 0;
        for j=1:N
            dTdt(i) = dTdt(i) + sin(thetas(j) - thetas(i));
        end
        dTdt(i) = omegas(i) + K/N * dTdt(i);
    end
end