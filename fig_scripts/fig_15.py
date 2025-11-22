import elechons.models.laplacian_noise as l

print('note this script puts the plots in plts/laplacian_noise_modelling/')

l.one_D_cov(length = 1000, samples = 5000, l=0.5, g=1, mean_of_means=10, var_of_means=2)
l.plot_two_D_cov(n = 50, samples = 5000, l=0.5, g=1)