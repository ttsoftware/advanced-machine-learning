function m = gp_mean(y, k, ks)
    m = ks' * pinv(k) * y;
end