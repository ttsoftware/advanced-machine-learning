function h = plot_normal(mu, Sigma, varargin)
  if (nargin < 2)
    error('plot_normal: two input arguments are required');
  end % if
  
  D = numel(mu);  
  if (size(Sigma, 1) ~= D || size(Sigma, 2) ~= D)
    error('plot_normal: size mismatch');
  end % if
  
  %%%%%%%%%%%%%%%%%
  %% 2D GAUSSIAN %%
  %%%%%%%%%%%%%%%%%
  if (D == 2)
    [V, l] = eig(Sigma);
    l = max(l, 0);
    t = linspace(0, 2*pi, 100)';
    xy = [cos(t), sin(t)].';
    Txy = bsxfun(@plus, V * sqrt(l) * xy, mu(:));
    
    hh = plot(Txy(1, :), Txy(2, :), varargin{:});
    if (nargout > 0)
      h = hh;
    end % if
    
  %%%%%%%%%%%%%%%%%
  %% 3D GAUSSIAN %%
  %%%%%%%%%%%%%%%%%
  elseif (D == 3)
    [V, l] = eig(Sigma);
    l = max(0, diag(l));
    r = sqrt(l);
    [X, Y, Z] = ellipsoid(0, 0, 0, r(1), r(2), r(3));
    XYZ = [X(:), Y(:), Z(:)];
    TXYZ = bsxfun(@plus, XYZ * V.', mu(:).');
    TX = reshape(TXYZ(:, 1), size(X));
    TY = reshape(TXYZ(:, 2), size(X));
    TZ = reshape(TXYZ(:, 3), size(X));
    hh = surfl(TX, TY, TZ);
    set(hh, 'edgecolor', 'none', 'facealpha', 0.5);
    if (nargout > 0)
      h = hh;
    end % if
    
  else
    error('plot_normal: only 2D and 3D Gaussians are supported');
  end % if
end % function

