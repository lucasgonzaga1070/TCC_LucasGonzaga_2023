from matplotlib import use

use('Agg')
use('TkAgg')

import numpy as np
import skimage.exposure
from skimage.morphology import *
from skimage.measure import label
from skimage.filters import threshold_otsu
import pandas as pd
import scipy
import pylidc as pl
from pylidc.utils import consensus
import matplotlib.pyplot as plt
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


def isola_pulmao_v2(img):
    index_meio_pulmao = int(img.shape[2]/2)
    lim = skimage.filters.threshold_multiotsu(img)

    img = (img > lim[0]) & (img < lim[1])
    labels = scipy.ndimage.label(img)

    val = labels[0][256, 128, index_meio_pulmao]
    out = labels[0] == val

    out = skimage.morphology.binary_closing(out, skimage.morphology.ball(10))
    out = out * 65535
    out = out.astype(np.uint16)

    return out


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
    hist = skimage.exposure.histogram(img)
    zeros = 0
    cont = 2

    lim = []
    while zeros < 2 and cont < hist[0].size:
        if hist[0][cont] > 5000 and hist[0][cont - 1] <= 5000 and zeros < 1:
            zeros += 1
            lim.append(cont)
        elif hist[0][cont] < 5000 and hist[0][cont - 1] >= 5000 and zeros >= 1:
            zeros += 1
            lim.append(cont)
        cont += 1

    img = skimage.exposure.rescale_intensity(img, in_range=(hist[1][lim[0]], hist[1][lim[1]]))
    plt.figure()
    plt.stem(hist[1][1:], hist[0][1:], markerfmt=".")
    plt.vlines(hist[1][lim[0]], ymin=0, ymax=hist[0][1:].max(), colors='r')
    plt.vlines(hist[1][lim[1]], ymin=0, ymax=hist[0][1:].max(), colors='r')
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

