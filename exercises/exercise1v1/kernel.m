%% function K = kernel(x, y, lambda, theta)
%%   Evaluate the squared exponential kernel function with parameters
%%   lambda and theta.
%%
%%   x and y should be NxD and MxD matrices. The resulting
%%   covariance matrix will be of size NxM.

function K = kernel(x, y, lambda, theta)
  D = pdist2(x, y, 'euclidean');
  K = theta .* exp(-0.5 .* D.^2 .* lambda);
end
