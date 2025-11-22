import elechons.regress_temp as r

r.init('mean')

print('note this script puts the plot in plts/autoregression_fit_imgs/')

r.plot_correlation_v_dist(r.regression_error)