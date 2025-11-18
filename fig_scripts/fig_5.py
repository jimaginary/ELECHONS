import scripts.DGFT_compression as dgft

for p in [50, 20, 15, 10, 7, 5, 3]:
    print(f'1/{p}')
    dgft.DGFT_compression(100-100/p, 8, 'mean')