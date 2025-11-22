import elechons.regress_temp as r

r.init('mean')

print('note this script generates both plots 16 and 17')

r.plot_kriging_from_cov_model(t=0, l=0.0019375)