clear all
close all

%% Load data
load clusterdata2d % gives 'data' -- also try with other datasets!
[N, D] = size(data);

%% Initialize parameters
K = 3; % try with different parameters
mu = cell(K, 1);
Sigma = cell(K, 1);
pi_k = ones(K, 1)/K;
for k = 1:K
  % Let mu_k be a random data point:
  mu{k} = data(randi(N), :);
  % Let Sigma_k be the identity matrix:
  Sigma{k} = eye(D);
end % for

%% Loop until you're happy
max_iter = 100; % XXX: you should find a better convergence check than a max iteration counter
log_likelihood = zeros(max_iter,1);
delta = 10000;
iter = 1;
while (delta > 0.1) && (iter < max_iter)  %% Compute responsibilities
  % XXX: FILL ME IN!
  gamma = zeros(K,N);
  for i=1:N
      for j= 1:K
         gmm = 0;
         for h = 1:K
             gmm = gmm + (pi_k(h,1) * mvnpdf(data(i,:), mu{h}, Sigma{h}));
         end
        gamma(j,i) = (pi_k(j,1) * mvnpdf(data(i,:), mu{j}, Sigma{j})) / gmm;
      end
  end
  
  %% Update parameters
  % N_k
  N_k = zeros(K,1);
    for k=1:K
      for n = 1:N      
          N_k(k,1) = N_k(k,1) + gamma(k,n);
      end
    end
  
  %Mu
  
  for k = 1:K
      acc = 0;
      for n=1:N,
          acc = acc + gamma(k,n) * data(n,:);
      end
      mu{k} = acc/N_k(k);
  end
  
  %Sigma
  for k=1:K
      acc = zeros(D);
      for n=1:N
          acc = acc + gamma(k,n)*((data(n,:) - mu{k})')*((data(n,:) - mu{k}));
      end
      Sigma{k} = acc/N_k(k);
  end
  
  %Pi
  for k=1:K
      pi_k(k,1) = N_k(k)/N;
  end
  
  %% Compute log-likelihood of data
  
  for n=1:N
      log_n = 0;
      for k=1:K
          log_n = log_n + pi_k(k) * mvnpdf(data(n,:), mu{k}, Sigma{k});
      end
      
      log_likelihood(iter) = log_likelihood(iter) + log(log_n);
  end  

  if ( iter > 1)
    delta = log_likelihood(iter) - log_likelihood(iter-1);
  end
  
  iter = iter + 1

  
end % for

%% Plot log-likelihood -- did we converge?
figure
plot(log_likelihood(1:(iter-1)));
xlabel('Iterations'); ylabel('Log-likelihood');

%% Plot data
figure
if (D == 2)
  plot(data(:, 1), data(:, 2), '.');
elseif (D == 3)
  plot3(data(:, 1), data(:, 2), data(:, 3), '.');
end % if
hold on
for k = 1:K
  plot_normal(mu{k}, Sigma{k});
end % for
hold off
