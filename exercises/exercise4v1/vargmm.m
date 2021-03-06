clear all
close all

%% Load data
load clusterdata2d % gives 'data'
[N, D] = size(data);

%% Wrapper for digamma function
digamma = @(x) psi(0, x);

%% Number of components/clusters
K = 10;

%% Priors
alpha0 = 1e-3; % Mixing prior (small number: let the data speak)
m0 = zeros(1, D); beta0 = 1e-3; % Gaussian mean prior
v0 = 3e1; W0 = eye(D)/v0; % Wishart covariance prior

%% Initialize parameters
m_k = cell(K, 1);
W_k = cell(K, 1);
beta_k = repmat(beta0 + N/K, [1, K]); % 1xK
alpha_k = repmat(alpha0 + N/K, [1, K]); % 1xK
v_k = repmat(v0 + N/K, [1, K]); % 1xK
for k = 1:K
    % Let m_k be a random data point:
    m_k{k} = data(randi(N), :); % 1xD
    % Let W_k be the mean of the Wishart prior:
    W_k{k} = v0*W0; % DxD
end % for
Nk = zeros(1, K) / K; % 1xK

%% Loop until you're happy
max_iter = 100;
for iter = 1:max_iter
    %% Variational E-step
    % XXX: FILL ME IN!
    
    p =zeros(N,K);
    r = zeros(N,K);
    for k = 1:K
        E_var = 0;
        for i=1:D
            E_var = E_var + digamma((v_k(k)+1-i)/2);
        end
        
        alpha_hat = sum(alpha_k);
        E_var = E_var + D * log(2) + logdet(W_k{k});
        E_pi = digamma(alpha_k(k)) - digamma(alpha_hat);
        
        for n = 1:N
            E_mu_lambda = D * pinv(beta_k(k)) + v_k(k) * (data(n,:) - m_k{k}) * W_k{k} * (data(n,:) - m_k{k})';
            p(n,k) = E_pi + 0.5 * E_var - (D/2) * log(2*pi) - 0.5 * E_mu_lambda;
        end
        
    end
    
    for n=1:N
        p_n = 0;
        for k=1:K
            p_n = p_n + exp(p(n,k) - max(p(n,:)));
        end
        
        for k=1:K
            r(n,k) = exp(p(n,k) - max(p(n,:))) / p_n;
        end
    end
    
    %% Variational M-step
    % XXX: FILL ME IN
    
    for k=1:K
        Nk(1,k) = 0;
        for n=1:N
            Nk(1,k) = Nk(1,k) + r(n,k);
        end
    end
    
    for k=1:K
        x_bar = zeros(1,D);
        for n=1:N
            x_bar = x_bar + r(n,k) * data(n,:);
        end
        x_bar = x_bar / Nk(1,k);
        
        S = zeros(D,D);
        for  n= 1:N
            S = S + r(n,k) * (data(n,:) - x_bar)' * (data(n,:) - x_bar);
        end
        S = S / Nk(1,k);
        
        alpha_k(k) = alpha0 + Nk(1,k);
        beta_k(k) = beta0 + Nk(1,k);
        m_k{k} = (beta0 * m0+ Nk(1,k) * x_bar)/beta_k(k);
        W_k{k} = pinv(pinv(W0) + Nk(1,k) * S + (beta0 * Nk(1,k))/(beta0 + Nk(1,k)) * (x_bar - m0)' * (x_bar - m0));
        v_k(k) = v0 + Nk(1,k);
    end
    
end % for

%% Plot data with distribution (we show expected distribution)
figure
plot(data(:, 1), data(:, 2), '.');
hold on
for k = 1:K
    if (Nk(k) > 0)
        plot_normal(m_k{k}, pinv(v_k(k) * (W_k{k})), 'linewidth', 2);
    end % if
end % for
hold off

%% Now, animate the uncertainty by sampling
%{
num_samples = 100;
figure
for s = 1:num_samples
  plot(data(:, 1), data(:, 2), '.');
  hold on
  for k = 1:K
    if (Nk(k) > 0)
      L = wishrnd(W_k{k}, v_k(k));
      Sigma = pinv(L);
      mu = mvnrnd(m_k{k}, Sigma/beta_k(k));
      plot_normal(mu, Sigma, 'linewidth', 2);
    end % if
  end % for
  hold off
  pause(0.1)
end % for

%% Animate the entire mixture distribution
figure
[X, Y] = meshgrid(linspace(-2, 2, 50));
XY = [X(:), Y(:)]; % 2500x2
for s = 1:num_samples
  pi_k = dirrnd(alpha_k);
  Z = zeros(size(X));
  for k = 1:K
    L = wishrnd(W_k{k}, v_k(k));
    Sigma = pinv(L);
    mu = mvnrnd(m_k{k}, Sigma/beta_k(k));
    Z(:) = Z(:) + pi_k(k) * mvnpdf(XY, mu, Sigma);
  end % for
  h = surfl(X, Y, Z); set(h, 'edgecolor', 'none');
  axis([-2, 2, -2, 2, 0, 0.3]);
  view(200*s/num_samples, 30);
  pause(0.1);
end % for
%}
