function p = gp_loglikelyhood(N, x, sigma2, lambda, theta)
    K = kernel(x, x, lambda, theta) + sigma2 * eye(N);
    p = ((-N)/2) * log(2*pi) - 0.5 * logdet(K) - 0.5 * y' * pinv(K) * y;
end