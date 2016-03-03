% function delta = subtract_orientation(theta1, theta2)
%   Subtract two orientations with values between -pi and pi.
%   The result is also between -pi and pi.
%
% See also: add_orientation

function delta = subtract_orientation(theta1, theta2)
  delta = -abs(theta1 - theta2);
  idx = (delta < pi);
  delta(idx) = delta(idx) + 2*pi;
end % function

