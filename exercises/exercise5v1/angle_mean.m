% function mu = angle_mean(angles, weights)
%   Compute the weights mean of a set of angles.

function mu = angle_mean(angles, weights)
  x = cos(angles(:)); % Nx1
  y = sin(angles(:)); % Nx1
  mux = weights(:).' * x; % 1x1
  muy = weights(:).' * y; % 1x1
  mu = atan2(muy, mux);
end % function
