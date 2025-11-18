import elechons.regress_temp as r

r.init('mean')

print('note this script generates all distribution hist and qq plots and puts them in plts/distribution_imgs')

r.save_hist_qq_subplots()