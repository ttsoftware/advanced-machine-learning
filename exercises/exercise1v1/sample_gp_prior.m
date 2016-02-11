clear all
close all

%% In this exercise, we will draw samples from a Gaussian process prior
%% with zero mean and a squared exponential covariance function.
%% The function 'kernel' (see 'kernel.m') implements the covariance function.
%% We will draw samples over the interval [-2, 2].
%% The kernel function has parameters lambda and theta (try experimenting with their values)
lambda = 10;
theta  = 2;

%% Define sample interval
M = 100; % dimensionality of samples -- increasing it implies a an increased resultion of the underlying function
xs = linspace(-2, 2, M).'; % Mx1

%% Evaluate the covariance matrix as k(xs, xs)
Sigma = kernel(xs, xs, lambda, theta);

%% Draw samples
S = 15; % number of samples that we draw
samples = sample_gaussian(zeros(M, 1), Sigma, S); % SxM

%% Plot samples
figure
hold all
for s = 1:S
  plot(xs, samples(s, :));  
end % for
hold off
title('Samples from GP')

%% Show covariance matrix
figure
imagesc(xs, xs, Sigma)
colorbar
title('Covariance matrix')
