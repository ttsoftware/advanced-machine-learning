function ld = logdet(M)
  if (false)
    ld = sum(log(diag(M)));
  else
    try
      C = chol(M);
      ld = 2 * sum(log(diag(C)));
    catch
      C = eig(M);
      C(C < 0) = 0;
      ld = sum(log(real(C)));
    end % try
  end % if
end % function
