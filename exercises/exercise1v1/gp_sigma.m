function s = gp_sigma(k, ks, kss)
    s = kss - ks' * pinv(k) * ks;
end