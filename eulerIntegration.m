function [T, Y] = eulerIntegration(dydt, T, y0)
    T = reshape(T, [numel(T), 1]);
    Y = zeros(numel(T), numel(y0));
    Y(1,:) = y0;
    for i=2:numel(T)
        dydtHere = dydt(T(i-1,:), Y(i-1,:));
        Y(i, :) = Y(i-1, :) + (T(i) - T(i-1)) * reshape(dydtHere, [1, numel(y0)]);
    end
end