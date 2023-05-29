from matplotlib import pyplot as plt
import cv2
import os
import numpy as np

def make_plot(rows: list, exp_id: str):
    fig, axs = plt.subplots(3,3)


    base_dir = 'YaleFaceDatabase'

    
    cols = ['centerlight', 'leftlight', 'rightlight']

    ax0 = 0
    for subject_id in rows:
        ax1 = 0

        for illumination in cols:
        
            b = plt.imread(os.path.join(base_dir, f'subject{subject_id}.{illumination}'))

            axs[ax0, ax1].imshow(b, 'gray')
            axs[ax0, ax1].axis('off')

            ax1 += 1

        ax0 += 1

    for ax, col in zip(axs[0], cols):
        ax.set_title(col)

    fig.tight_layout()
    plt.savefig(f'./images/face_experiment{exp_id}.pdf')

if __name__ == '__main__':
    make_plot(['01','02','03'], '123')

    make_plot(['05','06','07'], '567')
        
