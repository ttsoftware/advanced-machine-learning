%% function samples = sample_gaussian(mu, Sigma, S)
%%   Sample S samples from a D-dimensional Gaussian with mean
%%   mu and covariance matrix Sigma.
%%   The output is a SxD matrix.

function samples = sample_gaussian(mu, Sigma, S)
  if (nargin < 2)
    error('sample_gaussian: not enough input arguments');
  end % if
  if (nargin < 3)
    S = 1;
  end % if
  
  D = size(Sigma, 1);
  if (D ~= numel(mu))
    error('sample_gaussian: mean and covariance do not match in size');
  end % if
  
  [V, lambda] = eig(0.5*(Sigma+Sigma.'));
  lambda = real(diag(lambda));
  min_lambda = min(lambda);
  if (min_lambda < -1e-6)
    warning('Covariance matrix not positive definite: smallest eigenvalue is %f', min_lambda);
  end % if
  lambda(lambda < 0) = 0;
  A = V * diag(sqrt(lambda));
  
  S0 = randn(S, D);
  samples = bsxfun(@plus, S0 * A.', mu(:).');
end % function
