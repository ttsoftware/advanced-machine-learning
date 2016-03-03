% function theta = add_orientation(theta1, theta2)
%   Add two orientations with values between -pi and pi. The result is
%   also between -pi and pi.
%
% See also: subtract_orientation.
function theta = add_orientation(theta1, theta2)
  x = cos(theta1 + theta2);
  y = sin(theta1 + theta2);
  theta = atan2(y, x);
end % function

