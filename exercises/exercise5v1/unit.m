% function u = unit(v)
%   Convert a vector to a unit vector.

function u = unit(v)
  len = sqrt(sum(v.^2, 2));
  u = bsxfun(@times, v, 1./len);
end % function
