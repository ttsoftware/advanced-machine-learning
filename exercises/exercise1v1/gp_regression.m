clear all
close all

%% Load data
%% We subsample the data, which gives us N pairs of (x, y)
load weather
x = (1:20:1000)';
y = TMPMAX(x);
N = numel(y);

%% Standardize data to have zero mean and unit variance
x = (x - mean(x)) ./ std(x); % Nx1
y = (y - mean(y)) ./ std(y); % Nx1

%% We want to predict values at x_* (denoted xs in the code)
M = 1000;
xs = linspace(min(x), max(x), M).'; % Mx1
%xs = linspace(-2, 2, M).'; % Mx1 --- try predicting over this interval instead

%% Initial kernel parameters -- once you have a GP regressor,
%% you should play with these parameters and see what happens
lambda = 100;
theta  = 2;

%% Data is assumed to have variance sigma^2 -- what happens when you change this number?
sigma2 = (0.1).^2;

%% Compute covariance (aka "kernel") matrices
% XXX: FILL ME IN!
K   = kernel(x, x, lambda, theta) + sigma2 * eye(N); % NxN
Ks  = kernel(x, xs, lambda, theta); % NxM
Kss = kernel(xs, xs, lambda, theta); % MxM

%% Compute conditional mean p(y_* | x, y, x_*)
% XXX: FILL ME IN!
mu = gp_mean(y, K, Ks);
sigma = gp_sigma(K, Ks, Kss);

%% Plot the mean prediction
figure
plot(x, y, 'o-', 'markerfacecolor', 'k'); % raw data
hold all
plot(xs, mu); % mean prediction
hold off
title('Mean prediction');

%% Plot samples
figure
plot(x, y, 'ko', 'markerfacecolor', 'k'); % raw data
hold all
S = 50; % number of samples
samples = sample_gaussian(mu, sigma, S); % SxM
for s = 1:S
  plot(xs, samples(s, :));
end % for
hold off
title('Samples');

%% Evaluate log-likelihood for a range of lambda's
Q = 100;
possible_lambdas = linspace(1, 300, Q);
loglikelihood = NaN(Q, 1);
for k = 1:Q
  lambda_k = possible_lambdas(k);
  loglikelihood(k) = gp_loglikelyhood(N, x, sigma2, lambda_k, theta);
end % for
[~, idx] = max(loglikelihood);
lambda_opt = possible_lambdas(idx);
figure
plot(possible_lambdas, loglikelihood)
hold on
plot(possible_lambdas(idx), loglikelihood(idx), '*')
hold off
title('Log-likelihood for \lambda');
xlabel('\lambda')
