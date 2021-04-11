from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from glob import glob
import multiprocessing as mp
import skimage.morphology as morphology
import skimage.filters.rank as rank

def entropy(signal):
    '''
    function returns entropy of a signal
    signal must be a 1-D numpy array
    '''
    lensig = signal.size
    symset = list(set(signal))
    numsym = len(symset)
    propab = [np.size(signal[signal == i]) / (1.0 * lensig) for i in symset]
    ent = np.sum([p * np.log2(1.0 / p) for p in propab])
    return ent

'''
A jpeg image is loaded and a grey-scale copy of this image is generated. 
For further processing the PIL-images are converted to numpy-arrays.:
'''

def create_entropy_img(img_path_lst):

    save_path = "G:PythonData/ALUS/LightingTrial/results/"
    for img_path in img_path_lst:
        colorIm = Image.open(img_path)
        #colorIm = colorIm.resize((244, 244))
        greyIm = colorIm.convert('L')
        colorIm = np.array(colorIm)
        greyIm = np.array(greyIm)

        '''
        The parameter N defines the size of the region within which the entropy is calculated. 
        For N=5 the region contains 10*10=100 pixel-values. 
        In the following loop for each pixel position the corresponding neighbour region is extracted. 
        The 2-dimensional neighbourhood is flattened into a 1-dimensional numpy array and passed to the entropy function. 
        The entropy values are inserted into the entropy-array E.:
        '''

        N = 1
        S = greyIm.shape
        E = np.array(greyIm)
        ent_total = 0
        for row in tqdm(range(S[0])):
                for col in range(S[1]):
                        Lx = np.max([0, col-N])
                        Ux = np.min([S[1], col+N])
                        Ly = np.max([0, row-N])
                        Uy = np.min([S[0], row+N])
                        region = greyIm[Ly:Uy, Lx:Ux].flatten()
                        E[row, col] = entropy(region)
                        ent_total += E[row, col]

        print(img_path, ent_total)

        test_img = rank.entropy(greyIm, morphology.disk(2))

        plt.subplot(1, 3, 1)
        plt.imshow(colorIm)

        plt.subplot(1, 3, 2)
        plt.imshow(test_img)

        plt.subplot(1, 3, 3)
        # plt.imshow(E, cmap=plt.cm.jet)
        plt.imshow(E)
        # plt.xlabel('Entropy in 10x10\nneighbourhood')
        plt.colorbar()

        saved_file_name = img_path.split("\\")[-1].split(".")[0] + "_entropy_" + str(ent_total) + ".jpg"
        saved_file_path = save_path + saved_file_name

        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5)
        plt.savefig(saved_file_path, dpi=1200)
        plt.clf()


if __name__ == "__main__":
    img_dir = "G:PythonData/ALUS/20_11_18_LightingTrial/data/*"

    num_workers = 3
    jobs = []

    img_paths = glob(img_dir)
    mp_img_path_lsts = [img_paths[i::num_workers] for i in range(num_workers)]
    for img_path_lst in mp_img_path_lsts:
        p = mp.Process(target=create_entropy_img, args=(img_path_lst,))
        jobs.append(p)
        p.start()
