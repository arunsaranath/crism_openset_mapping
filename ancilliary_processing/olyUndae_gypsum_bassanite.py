# -*- coding: utf-8 -*-

"""
FileName:               olyUndae_gypsum_bassanite.py
Author Name:            Arun M Saranathan
Description:            This code file contains the code to map the subtle differences between gypsum and bassanite
                        in the CRISM images in olympia undae

Date Created:           01st March 2022
Last Modified:          01st March 2022
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
import math
from tqdm import tqdm
from spectral.io import envi
import os

from spectral_utilities.spectral_utilities import spectral_utilities


def getAngle(a, b, c):
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return ang + 360 if ang < 0 else ang


def map_gypsum_bassanite(img_name):
    hsi_img, hsi_header = spectral_utilities().open_hsiImage(img_name, strt_wvl=1.0275, stop_wvl=2.6)
    wvl = np.asarray(hsi_header['wavelength'])
    'Apply the smoothing kernel to the image'
    hsi_img = spectral_utilities().filter_hsi_bands(np.asarray(hsi_img))
    'Get the frame information'
    frm_info = spectral_utilities().extract_crismFrameInfo(hsi_img)

    'Also read in the best_guess_image'
    best_guess_img = img_name.replace('.img', '_smoothed5_CR_micaMaps_BestGuess.img')
    bg_img, bg_header = spectral_utilities().open_hsiImage(best_guess_img)
    bg_img = np.asarray(bg_img)
    'Use it to identify '
    bg_img[bg_img > 0] = 1
    bg_img = np.sum(bg_img, axis=2)

    decision_mat = np.zeros((hsi_img.shape[0], hsi_img.shape[1], 3))
    'Iterate over the valid pixels'
    for row in tqdm(range(frm_info['strtRow'], frm_info['stopRow'])):
        for col in range(frm_info['strtCol'], frm_info['stopCol']):
            # row = 137
            # col = 346
            if bg_img[row, col] == 1:
                chosen_spectrum = np.squeeze(hsi_img[row, col, :])

                'Get the hydration band feature'
                chosen_hyd_band = chosen_spectrum[130:145]
                min_pos = 130 + np.argmin(chosen_hyd_band)

                if np.abs(min_pos - 139) < np.abs(min_pos - 135):
                    decision_mat[row, col, 0] = 1
                    # print(f"Marked as gypsum, minimum at {wvl[min_pos]")
                else:
                    """a = np.array([wvl[135], chosen_spectrum[135]])
                    b = np.array([wvl[140], chosen_spectrum[140]])
                    c = np.array([wvl[144], chosen_spectrum[144]])

                    ba = a - b
                    bc = c - b

                    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                    angle = np.arccos(cosine_angle)
                    angle = np.degrees(angle)
                    if angle <= 179.3:
                        decision_mat[row, col, 1] = 1
                    else:
                        decision_mat[row, col, 2] = 1"""
                    rat_min = chosen_spectrum[139] / chosen_spectrum[min_pos]
                    if rat_min < 1.015:
                        decision_mat[row, col, 1] = 1
                    else:
                        decision_mat[row, col, 2] = 1

    out_name = best_guess_img.replace('.img', '_manual_sel.hdr')
    out_name = out_name.replace('.IMG', '.hdr')
    header = {"Description": "Additional ancilliary mapping of Bassanite and Gypsum for Olympia Undae"}
    header["lines"] = hsi_header["lines"]
    header["samples"] = hsi_header["samples"]
    header["bands"] = decision_mat.shape[2]
    header['band names'] = ['Gypsum', 'Bassanite+ Gypsum Mix', 'Bassanite']
    envi.save_image(out_name, decision_mat, dtype=np.float32, force=True,
                    interleave='bil', metadata=header)


def map_spectral_properties(img_name):
    hsi_img, hsi_header = spectral_utilities().open_hsiImage(img_name, strt_wvl=1.0275, stop_wvl=2.6)
    wvl = np.asarray(hsi_header['wavelength'])
    'Apply the smoothing kernel to the image'
    hsi_img = spectral_utilities().filter_hsi_bands(np.asarray(hsi_img))
    'Get the frame information'
    frm_info = spectral_utilities().extract_crismFrameInfo(hsi_img)

    'Also read in the best_guess_image'
    best_guess_img = img_name.replace('.img', '_smoothed5_CR_micaMaps_BestGuess.img')
    bg_img, bg_header = spectral_utilities().open_hsiImage(best_guess_img)
    bg_img = np.asarray(bg_img)
    'Use it to identify '
    bg_img[bg_img > 0] = 1
    bg_img = np.sum(bg_img, axis=2)

    decision_mat = np.zeros((hsi_img.shape[0], hsi_img.shape[1], 6))
    'Iterate over the valid pixels'
    for row in tqdm(range(frm_info['strtRow'], frm_info['stopRow'])):
        for col in range(frm_info['strtCol'], frm_info['stopCol']):
            # row = 137
            # col = 346
            if bg_img[row, col] == 1:
                chosen_spectrum = np.squeeze(hsi_img[row, col, :])
                'position of 1.4 band minimum'
                decision_mat[row, col, 0] = wvl[55 + np.argmin(chosen_spectrum[55:70])]
                'position of 1.7 band minimum'
                decision_mat[row, col, 1] = wvl[105 + np.argmin(chosen_spectrum[105:125])]
                'position of 1.9 band minimum'
                decision_mat[row, col, 2] = wvl[130 + np.argmin(chosen_spectrum[130:145])]
                'ratio of chosen 1.9 band elements'
                decision_mat[row, col, 3] = chosen_spectrum[139]/ chosen_spectrum[135]
                'band depth of feature at 2.2'
                b = (wvl[182] -wvl[164]) / (wvl[196] -wvl[164])
                Rs_ = ((1-b) *chosen_spectrum[164]) + (b* chosen_spectrum[196])
                decision_mat[row, col, 4] = 1 - (chosen_spectrum[182] / Rs_)
                'band depth of feature at 1.7'
                b1 = (wvl[113] - wvl[104]) / (wvl[126] - wvl[104])
                Rs1_ = ((1 - b1) * chosen_spectrum[104]) + (b1 * chosen_spectrum[126])
                decision_mat[row, col, 5] = 1 - (chosen_spectrum[113] / Rs1_)

    temp = decision_mat[:, :, 1]
    temp[temp >= 1.81] = 0
    decision_mat[:, :, 1] = temp

    out_name = best_guess_img.replace('.img', '_manual_sel_params.hdr')
    out_name = out_name.replace('.IMG', '.hdr')
    header = {"Description": "Additional ancilliary mapping of Bassanite and Gypsum for Olympia Undae"}
    header["lines"] = hsi_header["lines"]
    header["samples"] = hsi_header["samples"]
    header["bands"] = decision_mat.shape[2]
    header['band names'] = ['1.4 min', '1.7 min', '1.9 min', 'ratio 1.94125/1.191478', 'BD2212', 'BD1763']
    envi.save_image(out_name, decision_mat, dtype=np.float32, force=True,
                    interleave='bil', metadata=header)


def map_create_products(img_name):
    hsi_img, hsi_header = spectral_utilities().open_hsiImage(img_name, strt_wvl=1.0275, stop_wvl=2.6)
    wvl = np.asarray(hsi_header['wavelength'])
    'Apply the smoothing kernel to the image'
    hsi_img = spectral_utilities().filter_hsi_bands(np.asarray(hsi_img))
    'Get the frame information'
    frm_info = spectral_utilities().extract_crismFrameInfo(hsi_img)

    img_rgb = hsi_img[frm_info['strtRow']:frm_info['stopRow'], frm_info['strtCol']:frm_info['stopCol'],
              np.asarray(hsi_header['default bands'], dtype=int)]
    lim_mat = [[0.028, 0.061], [0.039, 0.072], [0.039, 0.069]]
    for ii in range(img_rgb.shape[2]):
        temp = np.squeeze(img_rgb[:, :, ii])
        temp[temp <= 0] = 0
        temp = (temp- lim_mat[ii][0]) / (lim_mat[ii][1] - lim_mat[ii][0])
        temp[temp >=1] =1
        temp[temp <= 0] = 0
        img_rgb[:, :, ii] = 1. * temp


    fig1, ax = plt.subplots(figsize=(5,5))
    ax.imshow(img_rgb, vmin=img_rgb.min(), vmax=img_rgb.max(), )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    out1 = img_name.replace('.img', '_rgb.png')
    fig1.savefig(out1, bbox_inches='tight')

    'Also read in the best_guess_image'
    best_guess_img = img_name.replace('.img', '_smoothed5_CR_micaMaps_BestGuess.img')
    bg_img, bg_header = spectral_utilities().open_hsiImage(best_guess_img)
    bg_img = np.asarray(bg_img[frm_info['strtRow']:frm_info['stopRow'], frm_info['strtCol']:frm_info['stopCol'], :])


    map_img = np.zeros((bg_img.shape[0], bg_img.shape[1], 3))
    map_img[:, :, 0] = bg_img[:, :, 3]
    map_img[:, :, 1] = bg_img[:, :, 2] + bg_img[:, :, 1]
    map_img[:, :, 2] = bg_img[:, :, 0]

    fig2, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(map_img)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    out2 = img_name.replace('.img', '_map.png')
    fig2.savefig(out2, bbox_inches='tight')


    'Get the manual param'
    best_guess_img_mp = img_name.replace('.img', '_smoothed5_CR_micaMaps_BestGuess_manual_sel_params.img')
    bg_img_mp, bg_header_mp = spectral_utilities().open_hsiImage(best_guess_img_mp)
    bg_img_mp = np.asarray(bg_img_mp[frm_info['strtRow']:frm_info['stopRow'], frm_info['strtCol']:frm_info['stopCol'], 3])
    bg_img_mp = 1./ bg_img_mp

    fig3, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(bg_img_mp, vmin=0.97, vmax=1.03, cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    out3 = img_name.replace('.img', '_rat.png')
    fig3.savefig(out3, bbox_inches='tight')

if __name__ == "__main__":
    'Set the image name'
    folder_loc = '/Volume1/data/CRISM/atmCorr_stable/olympia_undae/'
    img_string = '_nr_ds.img'
    'Read in the image'
    'Iterate over the files in the specified folder'
    for r, d, f in os.walk(folder_loc):
        for file in f:
            if file.find(img_string) != -1:
                'The image address is'
                img_address = os.path.join(r, file)
                """map_spectral_properties(img_address)
                map_gypsum_bassanite(img_address)"""
                map_create_products(img_address)




    print('finished')