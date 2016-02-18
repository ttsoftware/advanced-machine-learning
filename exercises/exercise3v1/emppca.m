clear all
close all

%% Load data
load mnist1 % gives 'data' -- also try with other datasets!
[N, D] = size(data);

%% Initialize parameters
W = rand(2,D);
sigma_2 = rand;
Z = rand(N,2);

E_z = zeros(N,2);
E_zz = cell(2,2);


mu = sum(data) ./ N;

%% Loop until you're happy
max_iter = 10; % XXX: you should find a better convergence check than a max iteration counter
iter = 1;
while (iter < max_iter)  %% Compute responsibilities
    M = sigma_2 * eye(2) + W*W';
    for n = 1:N
        E_z(n,:) = pinv(M)*W*(data(n,:) - mu)';
    end


    for n=1:N
        E_zz{n} = sigma_2 * pinv(M) + E_z(n,:)'*E_z(n,:);
    end

    acc1 = zeros(2,D);
    acc2 = zeros(2,D);
    for n=1:N
        acc1 = acc1 + (data(n,:) - mu) * E_z(n,:)';
        acc2 = acc2 + E_zz(n,:)';
    end
    W = acc1 * acc2';

    acc3 = 0;
    for n=1:N
       acc3 = acc3 + norm(data(n,:) - mu)^2 - 2 * E_z(n,:)' * W * (data(n,:)-mu)' + trace(E_zz{n} * (W) * W'); 
    end

    sigma_2 = acc3/(N*D);



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
