import skimage
from skimage import io
from skimage import color

import warnings
from multiprocessing import Pool


def generate_CH0_sample_worker(Args):
    img_fname_in = Args[0]
    img_fname_out = Args[1]

    # img_fname_in = os.path.join(Data_DIR, Dataset_Name, Dataset_Input, block) + '/' + block + 'CH0.tif'
    tmp_img = io.imread(img_fname_in)
    tmp_img = skimage.exposure.rescale_intensity(tmp_img, in_range='image', out_range='dtype')
    tmp_img_frame = skimage.color.gray2rgb(tmp_img)
    # img_fname_out = os.path.join(Data_DIR, Dataset_Name, Dataset_Output, block) + '/temp/CH0_sample.jpg'
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        io.imsave(img_fname_out, tmp_img_frame)


def generate_CH0_samples_master(INPUT_PATH, OUTPUT_PATH, DATASET, BLOCKS, microscope, CORES):
    """
        We create the parameter list: [Args1, Args2, Args3]
        For each Args, we have (img_fname_in, img_fname_out)

    """
    if microscope == 'zeiss':
        CH0_SAMPLE_ARGS_LIST = []
        for BID in BLOCKS:
            block = 's' + BID[1:]  # BID 1:4 generally, 2:4 specifically
            img_fname_in = INPUT_PATH + DATASET + '/' + DATASET + '_' + block + 't01c1_ORG.tif'
            img_fname_out = OUTPUT_PATH + DATASET + '/' + BID + '/temp/CH0_sample.jpg'
            CH0_SAMPLE_ARGS_LIST.append([img_fname_in, img_fname_out])

    pool = Pool(processes=CORES)
    pool.map(generate_CH0_sample_worker, CH0_SAMPLE_ARGS_LIST)
    pool.close()
