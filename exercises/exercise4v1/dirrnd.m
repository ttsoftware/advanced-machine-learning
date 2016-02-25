function r = dirrnd(alpha)
  p = length(alpha);
  r = gamrnd(alpha, 1);
  r = r / sum(r);
end % function
