%% Define distribution
mu = {[1, 1], [-4, 3]};
Sigma = {2*eye(2), 4*eye(2)};
pi_k = [0.3, 0.7];
K = 2;

%% Plot distribution
[X, Y] = meshgrid(linspace(-12, 7, 100), linspace(-6, 9, 100));
XY = [X(:), Y(:)];
Z = zeros(size(X));
for k = 1:K
  Z(:) = Z(:) + pi_k(k) * mvnpdf(XY, mu{k}, Sigma{k});
end % for
contour(X, Y, Z)

%% Sample from distribution
N = 500;
samples = NaN(N, 2);
for n = 1:N
   % Choose latent variable to know which Gaussian to sample from
   g = rand;
   if g < pi_k(1,1)
       k=1;
   else 
       k=2;
   end   
   
   % Sample from the choosen Gaussian
  samples(n, :) = sample_gaussian(mu{k}, Sigma{k}); % XXX: ACTUALLY DRAW A SAMPLE HERE!!!
end % for
hold on
plot(samples(:, 1), samples(:, 2), '.');
hold off

