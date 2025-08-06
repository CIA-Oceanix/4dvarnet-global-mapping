import torch
import xarray as xr
import kornia

def dog_kornia(x, sigma, l3_mask):
    B, C, H, W = x.shape
    k = int(4 * sigma + 1)
    if k % 2 == 0:
        k += 1
        
    x = torch.nan_to_num(x, nan=0.0)
    #x_masked = x * l3_mask
    mask_bool = l3_mask.bool()

    data_filtered = torch.where(mask_bool, kornia.filters.gaussian_blur2d(x, (k, k), (sigma, sigma), separable = False), torch.nan)
    mask_filtered = torch.where(mask_bool, kornia.filters.gaussian_blur2d(l3_mask, (k, k), (sigma, sigma), separable = False), torch.nan)
    
    data_filtered_normalized = data_filtered / (mask_filtered + 1e-6)
    
    kernel = kornia.filters.get_gaussian_kernel2d((k, k), (sigma, sigma)) 
    
    %matplotlib inline
    vmin, vmax = -0.01, 0.01
    fig, ax = plt.subplots(1, 3, figsize = (30, 30))
    ax[0].imshow(kernel[0])
    ax[0].set_title('Gaussian filter')
    
    data = x[0, 0]
    ax[1].imshow(data[250:350, 250:350], cmap = "coolwarm", vmin = vmin, vmax = vmax)
    ax[1].set_title('Input')
    
    ax[2].imshow(data_filtered_normalized[0, 0][250:350, 250:350], cmap = "coolwarm",  vmin = vmin, vmax = vmax)
    ax[2].set_title('Filtered masked normalized output')
    plt.show()

    return data_filtered_normalized

def consecutive_filtering_kornia(x, sigma=1.0, K = 2, l3_mask = []):
    g = [dog_kornia(x, sigma, l3_mask)]
    for k in range(1, K):
        g.append(dog_kornia(g[k - 1], sigma, l3_mask))
    return g


tgt = xr.open_dataset('/Odyssey/public/altimetry_traces/2010_2023/gridded/gridded_input.nc').sla_filtered
l3_mask = tgt.notnull().astype('float32').values

time_idx = 350
filtered_data_kornia = consecutive_filtering_kornia(torch.Tensor(tgt[time_idx].values).unsqueeze(0).unsqueeze(0), 2.5, 4, torch.Tensor(l3_mask[time_idx]).unsqueeze(0).unsqueeze(0))

DoG_kornia = torch.diff(torch.stack(filtered_data_kornia, 0).squeeze(), dim = 0)
