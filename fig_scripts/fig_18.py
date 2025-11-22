import elechons.regress_temp as r

r.init('mean')

print('note this script puts the plots 18 and 19 in plts/autoregression_fit_imgs/')

r.plot_correlation_v_dist_angle(r.regression_error)