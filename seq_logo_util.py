import matplotlib as mpl
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch
from matplotlib.font_manager import FontProperties

fp = FontProperties(family="Arial", weight="bold") 
globscale = 1.35
NUMBERS = { "1" : TextPath((-0.30, 0), "1", size=1, prop=fp),
            "3" : TextPath((-0.305, 0), "3", size=1, prop=fp),
            "5" : TextPath((-0.35, 0), "5", size=1, prop=fp),
            "7" : TextPath((-0.366, 0), "7", size=1, prop=fp),
            "9" : TextPath((-0.384, 0), "9", size=1, prop=fp),
            "11" : TextPath((-0.398, 0), "E", size=1, prop=fp) }

ALPHABETS = { "A" : TextPath((-0.305, 0), "A", size=1, prop=fp),
            "T" : TextPath((-0.35, 0), "T", size=1, prop=fp),
            "G" : TextPath((-0.366, 0), "G", size=1, prop=fp),
            "C" : TextPath((-0.384, 0), "C", size=1, prop=fp) }


COLOR_SCHEME = {'1': 'yellow',
                '3': 'orange', 
                '5': 'red', 
                '7': 'blue', 
                '9': 'navy',
                '11': 'darkgreen'}

COLOR_SCHEME_ALPHABET = {'A': 'darkgreen', 
                'T': 'red', 
                'G': 'orange', 
                'C': 'navy'}

def numberAt(number, x, y, yscale=1, ax=None):
    text = NUMBERS[number]

    t = mpl.transforms.Affine2D().scale(1*globscale, yscale*globscale) + \
        mpl.transforms.Affine2D().translate(x,y) + ax.transData
    p = PathPatch(text, lw=0, fc=COLOR_SCHEME[number],  transform=t)
    if ax != None:
        ax.add_artist(p)
    return p

def alphabetAt(letter, x, y, yscale=1, ax=None):
    text = ALPHABETS[letter]

    t = mpl.transforms.Affine2D().scale(1*globscale, yscale*globscale) + \
        mpl.transforms.Affine2D().translate(x,y) + ax.transData
    p = PathPatch(text, lw=0, fc=COLOR_SCHEME_ALPHABET[letter],  transform=t)
    if ax != None:
        ax.add_artist(p)
    return p
