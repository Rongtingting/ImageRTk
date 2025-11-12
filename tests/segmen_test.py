# %%
import imagertk

input_dir = "/mnt/nfs/home/rongtinghuang/Results/EC_MIBI_Data/Pembro/Rosetta_processing/Rosetta_Compensated_Images/2022-10-27T16-17-36_EC_Pembro_TMA2_C1_8/fov-1-scan-1/rescaled"
output_dir = "/mnt/nfs/home/rongtinghuang/Results/EC_MIBI_Data/Pembro/Segmentation_Results/test_mesmer"
print(1)
# %%
imagertk.seg.segmentation_mibi_mesmer(input_dir = input_dir, 
                             output_dir = output_dir,
                             nuclear_markers = ["Histone H3", "dsDNA"],
                             membrane_markers = ["CD45", "Vimentin-works", "Pan-Keratin", "CD163"],
                             pixel_size_um=0.78,    
                             maxima_threshold=0.075,
                             interior_threshold=0.20,
                             scale=False,
                             tag="test",
                             num_threads=8
                             )



# %%
