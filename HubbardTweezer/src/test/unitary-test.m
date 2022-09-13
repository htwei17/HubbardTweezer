% Create the problem structure.
n = 3;
manifold = unitaryfactory(n);
problem.M = manifold;

% Generate random problem data.
A = manifold.rand();

% Define the problem cost function and its Euclidean gradient.
problem.cost = @(x) real(trace((x - A)' * (x - A))) / 2;
problem.egrad = @(x) x - A; % notice the 'e' in 'egrad' for Euclidean

% Numerically check gradient consistency (optional).
checkgradient(problem);

% Solve.
[x, xcost, info, options] = trustregions(problem);

% Display some statistics.
figure;
semilogy([info.iter], [info.gradnorm], '.-');
xlabel('Iteration number');
ylabel('Norm of the gradient of f');
