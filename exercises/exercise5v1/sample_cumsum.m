%% function samples = sample_gaussian(mu, Sigma, S)
%%   Sample S samples from a D-dimensional Gaussian with mean
%%   mu and covariance matrix Sigma.
%%   The output is a SxD matrix.

function samples = sample_cumsum(weights, S)
  samples = zeros(1,S);
  for i=1:S
      z = rand();
      cs = cumsum(weights);
      samples(1,i) = find(cs >= z, 1);
  end
end % function
