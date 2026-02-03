import numpy as np
import skimage.exposure
from skimage.morphology import *
from skimage.measure import label
from skimage.filters import threshold_otsu
import pandas as pd
import scipy
import pylidc as pl
from pylidc.utils import consensus
from time import time


def afinidade(seed, pixel, h, it, mode=0, Adj=1):
    hom = np.exp(-0.5 * (((np.abs(seed - pixel) - h[0]) / h[1]) ** 2))
    inte = np.exp(-0.5 * (((0.5 * np.abs(seed + pixel) - it[0]) / it[1]) ** 2))

    if mode == 0:
        wi = 0.5
        wh = 1 - wi
        return Adj * (wh * hom + wi * inte), wh, wi
    if mode == 1:
        wi = inte / (hom + inte)
        wh = 1 - wi

        return Adj * (wh * hom + wi * inte), wh, wi


def avaliarSegmentacao(objSegmentado, goldSTD):
    intersec = objSegmentado * goldSTD
    area_intersec = intersec.sum()
    area_goldStandart = goldSTD.sum()
    area_objSegmentado = objSegmentado.sum()
    tam_imagem = goldSTD.size

    vp = (area_intersec / area_goldStandart) * 100
    fp = ((area_objSegmentado - area_intersec) / area_goldStandart) * 100
    fn = ((area_goldStandart - area_intersec) / area_goldStandart) * 100
    od = (200 * vp) / (2 * vp + fn + fp)
    Or = (100 * vp) / (vp + fp + fn)

    return [vp, fp, fn, od, Or]


def dist_coordenada(a, b):
    return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2 + (b[2] - a[2]) ** 2)


def _largest_connected_component(mask):
    labeled, num = scipy.ndimage.label(mask)
    if num == 0:
        return mask
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    largest = counts.argmax()
    return labeled == largest


def isola_pulmao_v2(img):
    lim = skimage.filters.threshold_multiotsu(img)

    img = (img > lim[0]) & (img < lim[1])
    out = _largest_connected_component(img)

    out = skimage.morphology.binary_closing(out, skimage.morphology.ball(10))
    out = out * 65535
    out = out.astype(np.uint16)

    return out


def normalize_intensity(img, low_percentile=1.0, high_percentile=99.0):
    low = np.percentile(img, low_percentile)
    high = np.percentile(img, high_percentile)
    if low == high:
        return img
    return skimage.exposure.rescale_intensity(img, in_range=(low, high))


def fuzzy_adpatado(img, gs, roi, sliceNod):
    Cmin, Cmax, Lmin, Lmax = roi
    C = Cmin + int((Cmax - Cmin) / 2)
    L = Lmin + int((Lmax - Lmin) / 2)
    M, N, O = img.shape

    gs_label = label(gs)
    gs_out = gs_label == gs_label[L, C, sliceNod]

    # plt.figure()
    # plt.imshow(img[:, :, 89], cmap="gray")

    # Pré segmentação do pulmão ----------------------------------------------------------------------------------------
    print("Pré segmentação do pulmão")
    s1 = time()
    lung = isola_pulmao_v2(img)
    lung = lung / lung.max()
    lung = lung.astype(float)
    f1 = time()
    print(f1 - s1)
    # plt.figure()
    # plt.imshow(lung[:, :, 89], cmap="gray")

    # img = lung * img

    # Criando matrizes auxiliares (conectividade local e global, afinidade e ex seeds) ---------------------------------
    print("Criando matrizes auxiliares (conectividade local e global, afinidade e ex seeds)")
    connect = np.zeros_like(img)
    afinit = np.zeros_like(img)
    pathConnect = np.zeros_like(img)
    exSeeds = np.zeros_like(img)

    # Ajuste de intesidade para melhorar contraste ---------------------------------------------------------------------
    print("Ajuste de intesidade para melhorar contraste")
    s2 = time()
    img = normalize_intensity(img)
    f2 = time()
    print(f2 - s2)
    # plt.figure()
    # plt.imshow(img[:, :, 89], cmap='gray')

    # Selecionando região de interesse ---------------------------------------------------------------------------------
    print("Selecionando região de interesse")
    s3 = time()
    seed0 = img[L, C, sliceNod]

    connect[L, C, sliceNod] = 1
    afinit[L, C, sliceNod] = 1
    exSeeds[L, C, sliceNod] = 1

    # Inicio da fila e calculo dos valores de media e desvio da intesidade e homogeneidade -----------------------------
    print("Inicio da fila e calculo dos valores de media e desvio da intesidade e homogeneidade")
    img_teste_prisma = img.copy()
    dimensao_maior = (Lmax - Lmin) if (Lmax - Lmin) > (Cmax - Cmin) else (Cmax - Cmin)
    altura_prisma = dimensao_maior // 4 - 1 if (dimensao_maior // 4 - 1) > 0 else 1

    Lminatual = Lmin
    Lmaxatual = Lmax
    Cminatual = Cmin
    Cmaxatual = Cmax

    list1 = []
    # lista_imgs = []
    for i in range(altura_prisma + 1):
        print(Lminatual, Lmaxatual)
        img_teste_prisma[Lminatual:Lmaxatual, Cminatual:Cmaxatual, sliceNod - i] = 1
        img_teste_prisma[Lminatual:Lmaxatual, Cminatual:Cmaxatual, sliceNod + i] = 1

        list1.append(img[Lminatual:Lmaxatual, Cminatual:Cmaxatual, sliceNod - i].flatten())
        list1.append(img[Lminatual:Lmaxatual, Cminatual:Cmaxatual, sliceNod + i].flatten())
        # lista_imgs.append(img_teste_prisma[:, :, sliceNod - i])
        # lista_imgs.append(img_teste_prisma[:, :, sliceNod + i])
        Lminatual += 1
        Lmaxatual -= 1
        Cminatual += 1
        Cmaxatual -= 1

    I = np.abs(np.concatenate(list1) + seed0) * 0.5
    H = np.abs(np.concatenate(list1) - seed0)
    mediaH = np.mean(H)
    stdH = np.std(H) + 1e-9

    mediaI = np.mean(I)
    stdI = np.std(I) + 1e-9
    cont = 0
    f3 = time()
    print(f3 - s3)
    # Iterações --------------------------------------------------------------------------------------------------------
    s4 = time()
    print("Iterações")
    lista_seeds = [(L, C, sliceNod)]
    lista_connect = [1]
    lista_dists = [0]
    fila_df = pd.DataFrame({'seed': lista_seeds, 'connect': lista_connect, 'dist': lista_dists})

    while fila_df.shape[0] != 0 and exSeeds.sum() < 15 * (Cmax - Cmin) * (Lmax - Lmin):
        fila_df = fila_df.sort_values(by=['connect', 'dist'], ascending=[False, True])
        lista_seeds = list(fila_df['seed'])
        lista_connect = list(fila_df['connect'])
        lista_dists = list(fila_df['dist'])

        seedAtual = lista_seeds.pop(0)
        # print(lista_connect[0])
        lista_connect.pop(0)
        lista_dists.pop(0)

        seed_x = seedAtual[0]
        seed_y = seedAtual[1]
        seed_z = seedAtual[2]

        pos = [(seed_x - 1, seed_y, seed_z),
               (seed_x + 1, seed_y, seed_z),

               (seed_x, seed_y + 1, seed_z),
               (seed_x, seed_y - 1, seed_z),

               (seed_x, seed_y, seed_z + 1),
               (seed_x, seed_y, seed_z - 1)]

        for i in pos:
            if (i[0] >= M) or (i[1] >= N) or (i[2] >= O) or (i[0] < 0) or (i[1] < 0) or (i[2] < 0):
                continue
            if exSeeds[i] != 1 and lung[i] == 1 and abs(sliceNod - i[2]) <= altura_prisma + 4:
                exSeeds[i] = 1
                ua, wh, wi = afinidade(img[seedAtual], img[i], (mediaH, stdH), (mediaI, stdI), mode=1)
                afinit[i] = ua
                uk = np.min([afinit[i], connect[seedAtual]])
                pathConnect[i] = uk
                mica = np.max([connect[i], pathConnect[i]])
                connect[i] = mica

                lista_seeds.append(i)
                lista_connect.append(mica)
                lista_dists.append(dist_coordenada((L, C, sliceNod), i))

            fila_df = pd.DataFrame({'seed': lista_seeds, 'connect': lista_connect, 'dist': lista_dists})
    f4 = time()
    print(f4 - s4)
    # Pós-processamento ------------------------------------------------------------------------------------------------
    print("Pós-processamento")
    # plt.figure()
    # plt.imshow(connect[:, :, 89], cmap='gray')
    s5 = time()
    imgPos_mask = img * connect
    # plt.figure()
    # plt.imshow(imgPos_mask[:, :, 89], cmap='gray')

    th = threshold_otsu(imgPos_mask)

    imgBin = imgPos_mask >= th
    # plt.figure()
    # plt.imshow(imgBin[:, :, 89], cmap='gray')

    imgBin = binary_closing(imgBin, ball(10))
    # plt.figure()
    # plt.imshow(imgBin[:, :, 89], cmap='gray')
    f5 = time()
    print(f5 - s5)
    print("Processo finalizado")

    return imgBin, gs_out


def create_GoldSTD(file):
    # Query for a scan, and convert it to an array volume.
    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == file).first()
    vol = scan.to_volume(verbose=False)

    if vol.min() < 0:
        vol = (vol - vol.min())/(vol.max() - vol.min())
    else:
        vol = (vol + vol.min()) / (vol.max() + vol.min())

    vol = vol*65535
    vol = vol.astype(np.uint16)
    boolean = np.zeros_like(vol)

    # Cluster the annotations for the scan, and grab one.
    nods = scan.cluster_annotations()

    for nod in nods:
        cmask, cbbox, masks = consensus(nod, clevel=0.5, pad=[(20, 20), (20, 20), (0, 0)])

        boolean[cbbox] = cmask*65535

    return vol, boolean, nods

