# -*- coding: utf-8 -*-

from zooniverse_cutter.zooniverse_cookiecutter import FitsCutter
from astropy import units as u

#PATHS
GALAXY='NGC7496'#'NGC0628'

DRIVE_PATH =
PROJECT_PATH = DRIVE_PATH +
JWST_PATH = DRIVE_PATH + GALAXY + '\\JWST\\'


jwst_770_filename = 'background_sub\\' + GALAXY.lower() + '_miri_lvl3_f770w_i2d.fits'
jwst_770_filepath = JWST_PATH + jwst_770_filename

muse_ha_filename = PROJECT_PATH + GALAXY + '_IMAGE_FOV_WFI_Hasub-dr2-nati.fits'
hst_b_filename  = DRIVE_PATH + GALAXY + '\\data1\\' + GALAXY.lower() + '_uvis_f438w_exp_drc_sci.fits'

#Running initialisation with JWST, MUSE Ha and HST b-band.
cuts = FitsCutter.from_fits(jwst_770_filepath, hdu_index_primary=1, gridding=[1,1] *u.arcmin, secondary_filenames=[muse_ha_filename, hst_b_filename], hdu_indices_secondaries=[1,0])

#testing that the cutouts look correct using region files: WORKS!
# cuts.save_as_ds9regions(JWST_PATH)

###
# Colorisation params. `rescalefn` is the stretch (i.e., `linear`, `log`,
# 'sqrt' etc.)
#`min_max` is the contast, the intensity scaling. `scaletype=abs' means
# the `min_max` are units of image. If `scaletype=perc' percentiles are used
# (between 0 and 100) and will occur on a per cutout basis, meaning that
# while the contrast will appear similar between images, the absolute
# minimum and maximum value will vary
# `color` is the color. Can be a simple string of the primary additive
# pr subtractive colors otherwise, needs to be a hex string or rgb list.
miri_770_color = {
    'rescalefn':'asinh',
    'min_max':[0.1,10],
    'color':'red',
    'scaletype':'abs',
}

muse_ha_color = {
    'rescalefn':'log',
    'min_max':[60,110000],
    'color':'green',
    'scaletype':'abs',
}

hst_b_color = {
    'rescalefn':'linear',
    'min_max':[0.005,0.06],
    'color':'blue',
    'scaletype':'abs',
}

#This gets the final colorized images using the above parameters. I
#have written it so that reprojection is only done once.
# This is SLOW :( !!
colored = cuts.get_ColoredCutouts(miri_770_color, [muse_ha_color, hst_b_color])

#Image saving as pngs. If cols are not entered, the last run
#colorization is saved
#might want to make the autosave name include what colors/params were used
#in the generation of the image

cuts.save_colored_cutout_as_imagefile(
    path=JWST_PATH + 'cutout_test_images\\',
    bands=['F770W', 'Ha', 'b438'],
    colored_cutouts_rgb_arrays=colored
)
