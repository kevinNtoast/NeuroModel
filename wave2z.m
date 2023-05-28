wv = 400:700;
wv_t = (wv - 550)/100;

out = GTC(wv_t);
GTC(1)
% plot(wv_t, out)


function g = GTC(s)
    g = 52.14 * exp(-0.5 * (s/14.73).^2);
end

function wavelength_to_z(w)

end    