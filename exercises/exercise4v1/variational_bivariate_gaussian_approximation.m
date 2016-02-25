clear all
close all

%% First, we define a bi-variate (2D) Gaussian
mu = [2; 3]; % 2x1
Lambda = [3, 1; 1, 2]; % inverse covariance matrix, 2x2
Sigma = inv(Lambda);

%% We want to approximate this bi-variate Gaussian with a factorized
%% distribution, i.e. N([x1; x2] | mu, Sigma) \approx p_1(x1) * p2(x2).
%% We apply the iterative updates in Eq. 10.12 - 10.15 in Bishops book
%% without using the closed-form solution (for illustration).
m1 = randn();
m2 = randn();

max_iter = 10;
for iter = 1:max_iter
  %% First we plot the true Gaussian distribution
  plot_normal(mu, Sigma);

  %% Then we plot the current estimate of the factorized distribution
  hold on
  plot_normal([m1; m2], diag([1./Lambda(1, 1), 1./Lambda(2, 2)]));
  hold off
  axis([-1, 4, -1, 5]);
  pause(1)

  %% Now we update the factorized estimate
  m1 = mu(1,1) - pinv(Lambda(1,1)) * Lambda(1,2) * (m2 - mu(2,1)); % XXX: FILL ME IN!
  m2 = mu(2,1) - pinv(Lambda(2,2)) * Lambda(2,1) * (m1 - mu(1,1)); % XXX: FILL ME IN!

  
end % for
