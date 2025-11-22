import elechons.regress_temp as r

r.init('mean')

print('note this script puts the plot in plts/autoregression_fit_imgs/')

r.plot_autoregression_partial_corrs(max_delay=5)