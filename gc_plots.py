from apply_gradcam import main as gc
import matplotlib.pyplot as plt
import sqlite3
import PIL
import numpy as np

# https://stackoverflow.com/questions/41793931/plotting-images-side-by-side-using-matplotlib
# https://stackoverflow.com/questions/25812255/row-and-column-headers-in-matplotlibs-subplots
def add_headers(
    fig,
    *,
    row_headers=None,
    col_headers=None,
    row_pad=1,
    col_pad=5,
    rotate_row_headers=True,
    **text_kwargs
):
    # Based on https://stackoverflow.com/a/25814386

    axes = fig.get_axes()

    for ax in axes:
        sbs = ax.get_subplotspec()

        # Putting headers on cols
        if (col_headers is not None) and sbs.is_first_row():
            ax.annotate(
                col_headers[sbs.colspan.start],
                xy=(0.5, 1),
                xytext=(0, col_pad),
                xycoords="axes fraction",
                textcoords="offset points",
                ha="center",
                va="baseline",
                **text_kwargs,
            )

        # Putting headers on rows
        if (row_headers is not None) and sbs.is_first_col():
            ax.annotate(
                row_headers[sbs.rowspan.start],
                xy=(0, 0.5),
                xytext=(-ax.yaxis.labelpad - row_pad, 0),
                xycoords=ax.yaxis.label,
                textcoords="offset points",
                ha="right",
                va="center",
                rotation=rotate_row_headers * 90,
                **text_kwargs,
            )



DBNAME = "database.db"
conn = sqlite3.connect(DBNAME)
c = conn.cursor()


styles_raw = c.execute("SELECT DISTINCT style FROM artworks where used = True").fetchall()

allinputimages = [] # 2d list of 3 images for each style
styles = [] # clean style list
for style in styles_raw:
    style = style[0]

    styles.append(style)

    listperstyle = c.execute("SELECT filename FROM artworks where style = '" + style +  "'and used = True LIMIT 3").fetchall()
    allinputimages.append(listperstyle)

allresimages = []
allrestitles = []
c = 0
for imagelist, style in zip(allinputimages, styles):
    c+= 1

    for img in imagelist:

        outimg = gc("E:/BELL/resized/" + img[0], "test1")
        # outimg = np.asarray(img)


        allresimages.append(outimg)
        allrestitles.append(style)

fig, axs = plt.subplots(5, 3, figsize=(5, 3))
axs = axs.flatten()
for img, ax in zip(allresimages, axs):
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])

add_headers(fig, row_headers=styles)
plt.show()









