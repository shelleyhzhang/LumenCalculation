import tifffile, os, datetime, time
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from affine_coreg_processing import clean_binary_masks
from skimage.transform import resize as resize_sk
            

def run_oct_frame_algos_on_complete_scan(input_data, model, batch_size=8, keys=None,
                                         params=None, bg=None, verbose=False):

    if params is None:
        params = generate_oct_input_params()
    if bg is None:
        bg = np.ones((ALINE_DEPTH,), 'float32') * 43.0

    all_results = {}

    results = model.oct_basics(params, input_data[:batch_size], bg)

    if verbose:
        for key in results.keys():
            print(key)

    for key in keys:
        all_results[key] = results[key].numpy().squeeze()

    for i in range(batch_size, len(input_data), batch_size):
        start_frame = i
        end_frame = np.minimum(len(input_data), start_frame + batch_size)
        results = model.oct_basics(params, input_data[start_frame:end_frame], bg)
        for key in keys:
            all_results[key] = np.r_[all_results[key], results[key].numpy()]

    return all_results

def generate_oct_input_params():

    params = np.zeros(10, NP_FLOAT)
    params[0] = 0  # imaging medium
    params[1] = 7.0  # field of view
    params[2] = 0

    return params

def load_log_background(file_name):

    bg = np.fromfile(file_name, dtype='float32')

    return bg


if __name__ == '__main__':
    NP_FLOAT = np.float32
    ALINE_DEPTH = 1024 
    N_CONTRAST = 1.458 #refraction index in Angiogram contrast
    PLT_OFFSET = 11 # in range of [0 511]
    FWR_ANGD = 76 #degree
    SHEATH_MASK_TH = 0.97
    FLAG_PRINT = 2 #>=1 print border pixel vs. frames >=2 print unet_sheath >=3 save combo polar and xy images
    FLAG_CLEAN_MASK = 1
    FLAG_WRITE_XLSX = 0
    outer_med = pd.DataFrame()
    outer_std = pd.DataFrame()
    inner_med = pd.DataFrame()
    inner_std = pd.DataFrame()
    lumen_rad = pd.DataFrame()
    lumen_mm2 = pd.DataFrame()
    lumen_area_new = pd.DataFrame()
    lumen_area_algo = pd.DataFrame()
    fault_noborder = np.zeros((6,10))
    fault_width = np.zeros((6,10))
    local_path = Path(r'C:\Users\szhang\Documents\Angio\FIH')
    xls_file_name = str(local_path) + '\sheath_cleanedmask_outring_all.xlsx'
    models_folder = Path(r'C:\Users\Documents\GitHub\Tensorflow_Saved_Models')
    oct_no_post_model_folder = models_folder / 'OCT_No_PostProc'
    
    for batch in np.arange(2,4):
        oct_directory = Path(r'TBD\FIH_Batch'+str(batch)+r'\SpectraData\Patients')
        for i, patient in enumerate(oct_directory.glob('1020*')):
            if i == 2: #to select certain patient
                print('patient #', i+1, ': ', end=" ")
                when = os.stat(patient).st_ctime
                year = str(datetime.datetime.fromtimestamp(when).year)[-2:]
                print(year)
                for j, pullback in enumerate(patient.glob('Pullbacks/' + year + '*_*')):
                    if True:#j == 6: #to select certain pullback
                        oct_file_name = Path(str(pullback) + '/' + 'OCT_Processing_input.data')

                        if oct_file_name.exists():
                            print('pullback #', j+1, ': ', pullback, end =" ")
                            bg_file_name = pullback / 'Log_Background.raw'
                            
                            if bg_file_name.exists():
                                bg_vector = load_log_background(bg_file_name)
                            else:
                                bg_vector = np.ones((ALINE_DEPTH,), 'float32') * 43.0
                            oct_raw = np.fromfile(oct_file_name, dtype=np.float32, offset=1024).reshape(-1, 500, ALINE_DEPTH)
                            bg_log = read_background(bg_file_name, log_data = False)
                            oct_log = log_transform(oct_raw, noise_dB = bg_log)
                            #check if the file is large enough, i.e. having enough frames. For 367 frames, it is 734,001KB, 751617024 bytes
                            if os.path.getsize(oct_file_name) > int(7e8):
                                # rsplit() method splits a string into a list, starting from the right.
                                savepath = local_path / str(patient).rsplit("\\",1)[-1] / str(pullback).rsplit("\\", 1)[-1]
                                savepath.mkdir(exist_ok=True, parents=True)
                                sheath_file_name = savepath / 'unet_sheath.tif'
                                iq_fname = pullback / 'oct_scan_output00_oct_image_quality.Algo_Data'
                                if sheath_file_name.exists():
                                    unet_sheath = tifffile.imread(sheath_file_name).astype('float32')
                                    print('loading unet_sheath.tif')
                                else:
                                    # run u-net
                                    #oct_no_post_model = tf.keras.models.load_model(str(oct_no_post_model_folder))
                                    oct_no_post_model = tf.saved_model.load(str(oct_no_post_model_folder))
                                    oct_input_params = generate_oct_input_params()
                                    
                                    unet_keys = ['oct_output01_unet']
                                    
                                    oct_unet_results = run_oct_frame_algos_on_complete_scan(oct_raw[..., np.newaxis], 
                                                                                            oct_no_post_model,
                                                                                            keys = unet_keys, 
                                                                                            params = oct_input_params,
                                                                                            bg = bg_vector,
                                                                                            batch_size = 8)

                                    # get sheath OD location, Gen-1 has dedicated sheat binary mask
                                    unet_sheath = oct_unet_results[unet_keys[0]][:,:,:,2] #3rd channel of UNet out of 5 in total
                                    tifffile.imwrite(sheath_file_name, unet_sheath)
                                    print('run and save sheath unet')
                                # use guidecath oct detection to determine frame where CAN entered the guidecath
                                oct_iq = read_oct_algodata(iq_fname)
                                gc_idx = np.where(oct_iq == -1)[0]
                                if len(gc_idx) > 0:
                                    first_gc_idx = np.min(gc_idx)
                                    n_oct_frames = first_gc_idx
                                else:
                                    first_gc_idx = oct_raw.shape[0]
                                    n_oct_frames = oct_raw.shape[0]
                                
                                unet_sheath = (unet_sheath > SHEATH_MASK_TH).astype('int32') 
                                # morphological, imopen, imclose
                                # opening is the dilation of the erosion,  removes small objects from the foreground (usually taken as the bright pixels) of an image, placing them in the background.
                                # closing is the erosion of the dilation, removes small holes in the foreground, changing small islands of background into foreground.

                                open_size = np.maximum(2, 4.0)
                                close_size = 0.0 #np.maximum(1, 1.0)
                                erode_size = 0.0 #np.maximum(1, 2.0)
                                if FLAG_CLEAN_MASK:
                                    unet_sheath = clean_binary_masks(unet_sheath, open_radius=open_size, close_radius=close_size, erode_radius=erode_size).astype('int32')


                                unet_shape = unet_sheath.shape #frames 367, A-lines 512, depth 512
                                # reading parameter data to obtain FOV
                                fov = np.fromfile(pullback / 'Image_Processing_Parameters.data', dtype=np.float32)[3]
                                # planar resolution [mm/pixel] is FOV [mm] divided by # of pixels
                                polar_res = fov / unet_shape[2] / N_CONTRAST #0.009524498635644939 mm/pixel ~=9.52um/pixel
                                oct_res = fov / ALINE_DEPTH / N_CONTRAST #0.004762249317822469 mm ~=4.76um/pixel
                                
                                #assign regions beyond certain pixel as impossible for sheat to exist.
                                #unet_sheath[:,:,round(0.50/polar_res):-1] = 0.0

                                
                                sheath_xy = scan_convert_pullback(unet_sheath)

                                oct_enhc = enhance_pullback_with_averaging(oct_log)
                                oct_xy = scan_convert_pullback(oct_enhc)
                                oct_shape = oct_enhc.shape
                                if FLAG_CLEAN_MASK & FLAG_PRINT >= 2:
                                    tifffile.imwrite(savepath / 'unet_sheath_cleanedmask2.tif', unet_sheath)                        
                                
                                # find maximum along depth axis
                                print('unet_sheath shape is', unet_sheath.shape) #367 is frame, 512 is alines, 512 is radial res
                                # stack along depth so sheath_border has size of (367, 512, 2) = (frames, A-lines, inner/outer)
                                sheath_border = np.dstack((np.diff(unet_sheath, axis = -1).argmax(axis=-1)+1, 
                                                        np.diff(unet_sheath, axis = -1).argmin(axis=-1))).astype('float')
                                                                            
                                # The nominal wall thickness is .0044" (112um), but the range could be  .0039" to .0049" (99 -124 um).  
                                # Keep in mind that the OCT and NIRS go through the wall at 76deg from longitudinal axis which would make the wall look thicker than it really is.
                                # sheath width range is 99 - 124 um times refraction index of 1.4 and divided by sin(76deg) is 142.8um, 178.9um
                                width_tol = np.array([140, 180])/polar_res
                                sheath_width = (sheath_border[...,1] - sheath_border[...,0])[...,np.newaxis]
                                print('sheath_border shape is', sheath_border.shape)

                                # find median along A-line axis
                                sheath_border_med = np.nanmedian(sheath_border, axis=1) # (367, 2)
                                sheath_border_std = np.nanstd(sheath_border, axis=1) #masked array instead of NaN

                                print('sheath_border_med shape is', sheath_border_med.shape)
                                if FLAG_PRINT >= 1:
                                    
                                    plt.figure()
                                    plt.plot(sheath_border_med[0:n_oct_frames,0], label = 'inner sheath ring')
                                    plt.legend()
                                    plt.plot(sheath_border_med[0:n_oct_frames,1], label = 'outer sheath ring')
                                    plt.legend()
                                    plt.ylim([0, 50])
                                    plt.ylabel('Median of Inner and Outer Sheath Edge')
                                    plt.xlabel('Frame # (367)')
                                    
                                    plt.title(str(patient).rsplit("\\",1)[-1] + ": pullback_"+ str(pullback).rsplit("\\", 1)[-1])
                                    plt.savefig(savepath / 'unet_sheath_cleanedmask_border_median_invessel.png')
                                    
                                    plt.figure()
                                    plt.plot(sheath_border_std[0:n_oct_frames,0], label = 'inner sheath ring')
                                    plt.plot(sheath_border_std[0:n_oct_frames,1], label = 'outer sheath ring')
                                    plt.ylim([0, 25])
                                    plt.ylabel('Standard Deviation of Outer Sheath Edge')
                                    plt.xlabel('Frame # (367)')
                                    plt.title(str(patient).rsplit("\\",1)[-1] + ": pullback_"+ str(pullback).rsplit("\\", 1)[-1])
                                    plt.savefig(savepath / 'unet_sheath_cleanedmask_border_std_invessel.png')
                                    print('Saved unet_sheath_border median and std figures')

                                #to reshape sheath border from (367, 512, 2) to (367, 500, 2)
                                #oct_enhc shape in (367,500,1024)
                                oct_sheath_polarcombo = np.repeat(oct_enhc.astype('uint8')[..., np.newaxis], 3, axis=-1) #367, 500, 1024, 3

                                sheath_border_reshape = np.zeros(([oct_shape[0], oct_shape[1], 2])) #(367, 500)
                                len_tmp = [unet_shape[1], oct_shape[1]]                            
                                conv_dp = oct_shape[2]/unet_shape[2]
                                #lumen boundary radial reading
                                lumen_area_fname = pullback / 'oct_output03_lumen_area_mm2.Algo_Data'
                                lumen_boundary_fname = pullback / 'oct_scan_output06_lumen_boundary_radial.Algo_Data'
                                lumen_area_mm2 = read_oct_algodata(lumen_area_fname)
                                lumen_boundary_radial = read_oct_algodata(lumen_boundary_fname) #range is [0,1023], shape is (367,500)

                                print(f"Begin to save combo tif stacks")
                                oct_sheath_xycombo = np.repeat(np.zeros(oct_xy.shape).astype('uint8')[...,np.newaxis], 3, axis=-1)
                                tic = time.perf_counter()
                                for f in range(len(sheath_border)):
                                    #interpolate inner border
                                    sheath_border_reshape[f,:,0] = np.interp(np.linspace(0, len_tmp[0]-1, num = len_tmp[1]), np.arange(len_tmp[0]), sheath_border[f,:,0]*conv_dp)
                                    #interpolate outer border
                                    sheath_border_reshape[f,:,1] = np.interp(np.linspace(0, len_tmp[0]-1, num = len_tmp[1]), np.arange(len_tmp[0]), sheath_border[f,:,1]*conv_dp)
                                    
                                    #if PRINT_FLAG == 3:

                                    #assign inner border index to green #np.array(np.NaN).round().astype(int) is -2147483648
                                    oct_sheath_polarcombo[f,np.arange(oct_shape[1])[...,np.newaxis].astype(int),sheath_border_reshape[f,:,0].round()[...,np.newaxis].astype(int)] = np.array([0, 255, 0], dtype='uint8')
                                    #assign outer border index to red
                                    oct_sheath_polarcombo[f,np.arange(oct_shape[1])[...,np.newaxis].astype(int),sheath_border_reshape[f,:,1].round()[...,np.newaxis].astype(int)] = np.array([255, 0, 0], dtype='uint8')
                                    #assign lumen boundary to yellow
                                    oct_sheath_polarcombo[f,np.arange(oct_shape[1])[...,np.newaxis].astype(int),lumen_boundary_radial[f,:].round()[...,np.newaxis].astype(int) ] = np.array([255, 255, 0], dtype='uint8')

                                    #overlay inner border median 
                                    midline = np.arange(round(oct_shape[1]/3), round(oct_shape[1]*2/3),1)
                                    oct_sheath_polarcombo[f,midline[...,np.newaxis].astype(int),(np.ones((1,oct_shape[1],1,1))*sheath_border_med[f,0]*conv_dp).astype('int')] = np.array([0, 255, 0], dtype='uint8')
                                    #overlay outer border median
                                    oct_sheath_polarcombo[f,midline[...,np.newaxis].astype(int),(np.ones((1,oct_shape[1],1,1))*sheath_border_med[f,1]*conv_dp).astype('int')] = np.array([255, 0, 0], dtype='uint8')
                                    #overlay physical sheath borders
                                    
                                    phy_outer = np.arange(round(0.426/oct_res), round(0.447/oct_res),1)
                                    oct_sheath_polarcombo[f,midline[...,np.newaxis].astype(int),phy_outer[np.newaxis,...,np.newaxis, np.newaxis].astype('int')] = np.array([255, 255, 255], dtype='uint8')                                
                                    #write polar combo into tiff stack

                                if FLAG_PRINT == 3:
                                    tifffile.imwrite(savepath / 'oct_sheath_border_polar.tif', oct_sheath_polarcombo, photometric='rgb')     
                                    for color in np.arange(3):
                                        oct_sheath_xycombo[...,color] = scan_convert_pullback(oct_sheath_polarcombo[...,color])
                                    tifffile.imwrite(savepath / 'oct_sheath_border_xy.tif', oct_sheath_xycombo, photometric='rgb')                            
                                    toc = time.perf_counter()
                                    print(f"Saved combo tif stacks in {toc - tic:0.4f} seconds")

                                #to append inner sheath border median
                                header = pd.Series([str(patient).rsplit("\\", 1)[-1],str(pullback).rsplit("\\", 1)[-1], first_gc_idx])
                                to_append = pd.concat([header, pd.Series(sheath_border_med[...,0])])
                                inner_med = pd.concat([inner_med, to_append], axis=1)
                                #to append outer sheath border median
                                del to_append
                                to_append = pd.concat([header, pd.Series(sheath_border_med[...,1])])
                                outer_med = pd.concat([outer_med, to_append], axis=1)

                                #to append inner sheath border standard deviation
                                del to_append
                                to_append = pd.concat([header, pd.Series(sheath_border_std[...,0])])
                                inner_std = pd.concat([inner_std, to_append], axis=1)
                                #to append outer sheath border standard deviation
                                del to_append
                                to_append = pd.concat([header, pd.Series(sheath_border_std[...,1])])
                                outer_std = pd.concat([outer_std, to_append], axis=1)

                                #calculate lumen area  = wedge angle * 0.5 * (radius plus 0.42mm minus sheath border median as offset)^2
                                
                                #OD / (sin(beam angle) * 2) where OD is .0333" (846um), beam angle is 76° (1.326 radians). These are nominals.
                                #The tolerance on the beam angle is ± 2degrees. The answer is 436um.
                                fwr_factor = np.sin(np.radians(FWR_ANGD)) #0.9702957
                                rsq_c = 0.5*pow((lumen_boundary_radial*oct_res + 0.436 - oct_res*sheath_border_reshape[...,1])*fwr_factor, 2)
                                lumen_area_calc = np.sum(2*np.pi/len_tmp[1]*rsq_c, axis = -1)
                                lumen_area_calc[first_gc_idx:-1] = 0
                                #PLT offset set to constant at 22 for dimension of 1024
                                rsq_a = 0.5*pow((lumen_boundary_radial + PLT_OFFSET*conv_dp)*oct_res, 2)
                                lumen_area_old = np.sum(2*np.pi/len_tmp[1]*rsq_a, axis = -1)
                                lumen_area_old[first_gc_idx:-1] = 0
                                #print(lumen_boundary_radial.shape, lumen_area_mm2.shape, lumen_area_calc.shape)
                                #to append lumen radius
                                del to_append
                                to_append = pd.concat([header, pd.Series(np.max(lumen_boundary_radial, axis=1))])
                                lumen_rad = pd.concat([lumen_rad, to_append], axis=1)
                                
                                #to append lumen area (existing)
                                del to_append
                                to_append = pd.concat([header, pd.Series(lumen_area_mm2)])
                                lumen_mm2 = pd.concat([lumen_mm2, to_append], axis=1)

                                #to append lumen area (existing_calculation)
                                del to_append
                                to_append = pd.concat([header, pd.Series(lumen_area_old)])
                                lumen_area_algo = pd.concat([lumen_area_algo, to_append], axis=1)

                                del to_append
                                to_append = pd.concat([header, pd.Series(lumen_area_calc)])
                                lumen_area_new = pd.concat([lumen_area_new, to_append], axis=1)

    if FLAG_WRITE_XLSX:
        with pd.ExcelWriter(xls_file_name, mode="a", if_sheet_exists="overlay", engine="openpyxl") as writer:          
            outer_med.to_excel(writer, sheet_name='outer_med')   
            outer_std.to_excel(writer, sheet_name='outer_std') 
            inner_med.to_excel(writer, sheet_name='inner_med')   
            inner_std.to_excel(writer, sheet_name='inner_std')
            lumen_rad.to_excel(writer, sheet_name='lumen_radius_median') 
            lumen_mm2.to_excel(writer, sheet_name='lumen_area_mm2')
            lumen_area_algo.to_excel(writer, sheet_name='lumen_area_algo')
            lumen_area_new.to_excel(writer, sheet_name='lumen_area_new')