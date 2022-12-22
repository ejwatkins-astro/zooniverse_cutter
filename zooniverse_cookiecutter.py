# -*- coding: utf-8 -*-
"""
@author: Elizabeth Jayne Watkins

TODOS:
    [ ] - Test all the methods work;
    [ ] - More functionality for setting filenames for saves;
    [ ] - Fill in doc strings;
    [ ] - Functionality for flipping and rotating images;



"""

import functools
import os

from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy import wcs
from astropy.io import fits
from astropy.table import Table, QTable
from astropy.nddata import Cutout2D
from regions import RectangleSkyRegion, Regions
import reproject as rpj

import numpy as np
import matplotlib.pyplot as plt

import multicolorfits as mcf #Slightly modified version of this package

from astropy.nddata.utils import NoOverlapError

import copy


def get_fits_file_info(fits_file_name, extention=0, *header_info) :

    """Parameters: String
                *arg strings \n
    Returns: 2D  Numpy array, *args mulitple type, mainly Floats
    ----------------
    This function opens up fits files into a 2D Numpy array and loads the
    header information. If only specific header information is wanted, this
    function accepts args as strings to output the corresponding header
    infomation
    """

    with fits.open(fits_file_name) as hdulist:     # importing fits file

        data = hdulist[extention].data
        #print(repr(hdulist[0].header))
        if header_info == ():
            header_info_array = hdulist[extention].header

        else:
            header_info_array = []
            for i in range( len(header_info) ):
                header_info_array.append( hdulist[extention].header[header_info[i]])
        hdulist.close()

    return data, header_info_array

def get_perc_nan(image):
    """Finds out how much of the image is filled with NaNs
    """
    nans = np.isnan(image)
    perc = len(image[nans])/len(image.flatten()) * 100

    return perc


def zooniverse_cutout_filename_generator(center_ra_dec, boxsize_arcmin, band=None):
    """The filename template used to save data.
    """
    template = '%.4f_%.4f_%.1farcmin_%.1farcmin'

    filename = template % (center_ra_dec[0], center_ra_dec[1], boxsize_arcmin[0].value, boxsize_arcmin[1].value)

    if band is not None:
        filename = band + '_' + filename

    return filename

def update_hdu(hdu, data, header):
    """Updates the data and header in a hdu
    WARNING. Memory points at the same reference so
    if you want to preserve the origional hdu, enter
    hdu.copy() (or at some point this can be added to this function)?
    """
    hdu = hdu.copy()
    hdu.data = data
    hdu.header.update(header)

    return hdu

def save_cutout_as_fits(name_of_file, cutout_obj, hdu_obj_orig, file_type='.fits'):

    """
    This function saves astropy.nddata.Cutout2d objects to fitsfiles.
    """
    # cutout_data, cutout_header = get_cutout_data_header(cutout)

    hdu_obj_orig = update_hdu(hdu_obj_orig, *get_cutout_data_header(cutout_obj))
    hdu_obj_orig.header.update(cutout_header)

    if name_of_file[-5:] != file_type:
        name_of_file += file_type

    save_fitsfile(name_of_file, hdu_obj_orig)

def save_fitsfile(name_of_file, hdu, end='.fits'):
    """Using the filepath+filename and hdu object, saves a fits file
    """
    if name_of_file[-5:] != end:
        name_of_file = name_of_file + end
    hdu.writeto(name_of_file, overwrite=True)


def get_cutout_data_header(cutout):
    """gets the data and header from an astropy.nddata.Cutout2d object
    """
    cutout_data = cutout.data
    cutout_header = cutout.wcs.to_header()

    return cutout_data, cutout_header


def half_step_overlap_grid(image_shape, initial_grid_pixel):
    """Gridding method for the cutouts. Overlaps each cutout by 50%
    """

    # if image_shape[0] % initial_grid[0] !=0 or image_shape[1] % initial_grid[1] !=0:
    #     print('Grid does not divide equally into one of the axis')
    # else:

    # indicies = []
    y_step= initial_grid_pixel[0]/2
    y_inds = np.arange(y_step, image_shape[0], y_step, dtype=int)

    x_step= initial_grid_pixel[1]/2
    x_inds = np.arange(x_step, image_shape[1], x_step, dtype=int)

    return x_inds, y_inds


def convert_pixel2world(paired_pixel_coordinates, header):

    """
    Load the WCS information from a fits header, and use it
    to convert world coordinates to pixel coordinates.
    """
    w = wcs.WCS(header)
    paired_world_coordinates = w.all_pix2world(paired_pixel_coordinates, 0)
    return paired_world_coordinates

def convert_world2pixel(paired_world_coordinates, header):

    """
    Load the WCS information from a fits header, and use it
    to convert pixel coordinates to world coordinates.
    """
    w = wcs.WCS(header)
    paired_pixel_coordinates= w.all_world2pix(paired_world_coordinates, 0)
    return paired_pixel_coordinates

def get_pixelsize_hdu(hdu):
    """ Gets the pixel size (in degrees) from hdu
    """
    return get_pixelsize_header(hdu.header)

def get_pixelsize_header(header):
    """ Gets the pixel size (in degrees) from header
    """
    try:
        pixelsize = header['CDELT2']
    except KeyError:
        pixelsize = header['CD2_2']
    return pixelsize

def where_highest_res_list(hdus):
    """ Finds which hdu/fits image from a 2d array of hdus
    (per row) is at the highest resolution
    """
    row, col = np.shape(hdus)
    res = np.zeros([row, col])
    for i in range(row):
        for j in range(col):
            res[i,j] = get_pixelsize_hdu(hdus[i,j])
    highest_res_pos = np.nanargmin(res, axis=1)
    return highest_res_pos

def check_projection(hdu1, hdu2):
    """ Function not used in the end
    Checks if two headers are identical so that future functions
    can choose not to reproject
    """

    header1 = hdu1.header
    header2 = hdu2.header

    naxis1 = header1['NAXIS1'] == header2['NAXIS1']
    naxis2 = header1['NAXIS2'] == header2['NAXIS2']

    cdelt2 = get_pixelsize_hdu(hdu1) == get_pixelsize_hdu(hdu2)
    crpix1 = header1['CRPIX1'] == header2['CRPIX1']
    crpix2 = header1['CRPIX2'] == header2['CRPIX2']

    crval1 = header1['CRVAL1'] == header2['CRVAL1']
    crval2 = header1['CRVAL2'] == header2['CRVAL2']

    return all([naxis1, naxis2, cdelt2, crpix1, crpix2, crval1, crval2])

def make_header_3d(header):
    """Makes a 2d header 3d. BUG TODO: `header['NAXIS3']` assumes three
        images but depending on how its used, could have more or less than
        three images
    """
    header['NAXIS'] = 3
    header['NAXIS3'] = 3
    header['CDELT3'] = 1
    header['CRPIX3'] = 1
    header['CRVAL3'] = 1
    return header

def get_box_ds9_region(center_ra_dec_deg, size_wid_hei_arcmin, angle_deg=0):
    """Turns the cutout gridding into box ds9 regions
    """

    center_ra_dec_deg = SkyCoord(center_ra_dec_deg[0], center_ra_dec_deg[1], unit='deg', frame='fk5')

    if not isinstance(size_wid_hei_arcmin, u.quantity.Quantity):
        size_wid_hei_arcmin *= u.arcmin

    if not isinstance(angle_deg, u.quantity.Quantity):
        angle_deg *= u.deg

    # center_sky = SkyCoord(42, 43, unit='deg', frame='fk5')
    region_sky = RectangleSkyRegion(center=center_ra_dec_deg,
                                    width=size_wid_hei_arcmin[0],
                                    height=size_wid_hei_arcmin[0],
                                    angle=angle_deg
                                    )
    return region_sky

def stack_hdus(prime_cuts_hdu, secondary_cuts_hdu, where_seconary_overlap):
    """ Takes lists of hdu objects and combines them into a 2d array
    columns represent the hdu lists
    """

    if secondary_cuts_hdu is None:
        num_sec = 0
    else:
        secondary_cuts_hdu = np.atleast_1d(secondary_cuts_hdu)
        num_sec = len(secondary_cuts_hdu)

    hdus = np.zeros([len(prime_cuts_hdu),1+num_sec], dtype=object) * np.nan
    hdus[:,0] = prime_cuts_hdu
    for i in range(num_sec):
        if where_seconary_overlap is None:
            if len(prime_cuts_hdu) == len(secondary_cuts_hdu[i]):
                hdus[:,i+1] = secondary_cuts_hdu[i]

            else:
                raise IndexError('`secondary_cuts_hdu` is not the '\
                                 'same length as `prime_cuts_hdu` and '\
                                 '`where_seconary_overlap` is `None`')
        else:
            if len(where_seconary_overlap[i]) == len(secondary_cuts_hdu[i]):

                hdus[:,i+1] = secondary_cuts_hdu[i]
            elif len(prime_cuts_hdu) == len(secondary_cuts_hdu[i]):
                hdus[where_seconary_overlap[i],i+1] = secondary_cuts_hdu[i]
            else:
                raise IndexError('Where overlaps between primary and secondary '\
                                 'are incompatible')
    return hdus

def save_image(filename, arr, format='png', **kwargs):
    """wrapper for plt.imsave since the format doesn't
    automatically apply
    """
    format = kwargs.pop('format', format)
    if '.' in filename[-5:]:
        pass
    else:
        filename += '.' + format

    plt.imsave(filename, arr, format=format, **kwargs)

#Decorator than removes extra dimentsions
def correct_dimension(func):
    def wrapper(self, *args, **kwargs):
        output = func(self, *args, **kwargs)
        output_shape = np.shape(output)
        if len(output_shape) >= 3 and output_shape[0] == 1:
            return output[0]
        else:
            return output
    return wrapper

#Some dictionaries for colours
colors_rgb = {
    'red':[255,0,0],
    'yellow':[255,255,0],
    'green':[0,255,0],
    'cyan':[0,255,255],
    'blue':[0,0,255],
    'magenta':[255,0,255],
}

colors_hex = {
    'red':'#FF0000',
    'yellow':'#FFFF00',
    'green':'#00FF00',
    'cyan':'#00FFFF',
    'blue':'#0000FF',
    'magenta':'FF00FF',
}

# colors = {'hex':colors_hex,
#           'rgb':colors_rgb}



class ColorImages(object):
    """Class takes in an image (or list of images) and colorizes them
    """

    def __init__(self, images, rescalefn, min_max, color=None, scaletype='abs', colortype='hex'):

        self.method_colors = ['red', 'yellow', 'green', 'cyan', 'blue', 'magenta', 'grey', 'gray']

        if len(np.shape(images)) !=3:
            images = [images]
        self.images = images
        self.rescalefn = rescalefn
        self.min_max = min_max
        self.scaletype = scaletype
        self.colortype = colortype
        if color is not None:
            self.colored = self.get_color(color, self.colortype)
        else:
            self.colored=None

    def grey(self):
        grey_image = [mcf.greyRGBize_image(image, self.rescalefn, self.scaletype, self.min_max, gamma=2.2, checkscale=False) for image in self.images]

        return grey_image

    def gray(self):
        return self.grey()

    def red(self):
        return self._color(colors_hex['red'], 'hex')

    def yellow(self):
        return self._color(colors_hex['yellow'], 'hex')

    def green(self):
        return self._color(colors_hex['green'], 'hex')

    def cyan(self):
        return self._color(colors_hex['cyan'], 'hex')

    def blue(self):
        return self._color(colors_hex['blue'], 'hex')

    def magenta(self):
        return self._color(colors_hex['magenta'], 'hex')

    def _color(self, color, colortype=None):
        if colortype is None:
            colortype = self.colortype
        grey_images = self.grey()
        cols = [mcf.colorize_image(grey_image, color, colorintype=colortype, gammacorr_color=2.2) for grey_image in grey_images]

        return cols

    def _get_color(self, color, colortype=None):
        if colortype is None:
            colortype = self.colortype

        if color in self.method_colors:
            return getattr(self, color)()
        else:
            return self._color(color, colortype)

    @correct_dimension
    def get_color(self, color, colortype=None):
        if colortype is None:
            colortype = self.colortype
        return self._get_color(color, colortype)

    @correct_dimension
    def colorized_images(self, *img_list):

        if not isinstance(img_list[0], list):
            for i in range(len(img_list)):
                img_list[i] = [img_list[i]]

        final_images = []
        for i in range(len(img_list[0])):

            final_image = []
            for j in range(len(img_list)):
                final_image.append(img_list[j][i])
            final_images.append(mcf.combine_multicolor(final_image, gamma=2.2))

        return final_images



class ReprojectCutouts(object):
    """Takes cutout hdus and reprojects them the the highest resolution
    image
    """

    def __init__(self, prime_cuts_hdu, secondary_cuts_hdu, where_secondary_overlap=None):

        hdus = stack_hdus(prime_cuts_hdu, secondary_cuts_hdu, where_secondary_overlap)
        self.highest_res = where_highest_res_list(hdus)
        self.hdus_repr = self.reproject(hdus)

    @classmethod
    def from_FitsCutter(cls, FitsCutter_obj):

        prime_hdus, secondaries_hdus = [FitsCutter_obj.get_cutout_hdus()]
        where_seconary_overlaps = FitsCutter_obj.where_secondary_overlap

        cls(prime_hdus, secondaries_hdus, where_secondary_overlap)

    @classmethod
    def from_full_files(cls, primary_filename, gridding, prime_color_params, secondary_filenames=None, hdu_index_primary=0, hdu_indices_secondaries=None, grid_method=half_step_overlap_grid, secondary_colors_params=None):

        return ReprojectCutouts.from_FitsCutter(FitsCutter.from_fits(primary_filename, gridding, secondary_filenames, hdu_index_primary, hdu_indices_secondaries, grid_method))


    def reproject(self, stacked_hdus, method=rpj.reproject_interp):
        # high_hdu = stacked_hdus[:, self.highest_res]
        # other_hdu = np.delete(stacked_hdus, self.highest_res)
        num_cutouts, num_hdu = np.shape(stacked_hdus)

        for i in range(num_cutouts):

            res_ind = self.highest_res[i]
            cutout_row = stacked_hdus[i]

            highest_res_cutout_hdu = cutout_row[res_ind]
            highest_res_header = highest_res_cutout_hdu.header

            inds_update = np.delete(range(num_hdu), res_ind)
            imgs = []
            for n in inds_update:
                # try:
                if cutout_row[n] is None:
                    img = highest_res_cutout_hdu.data * np.nan
                else:
                    img = method(cutout_row[n], highest_res_header)[0]
                # except TypeError:
                #     img = highest_res_cutout_hdu.data * np.nan

                stacked_hdus[i,n] = update_hdu(copy.copy(highest_res_cutout_hdu.copy()), img, highest_res_header)
            # print(img)
        return stacked_hdus

class ColoredCutouts(object):
    """Takes in hdus of images (uually from cutouts), preps them for
    colorization by reprojection, and applies the colors via the params
    given.
    """

    def __init__(self, prime_cuts_hdu, prime_color_params, secondary_cuts_hdu=None, secondary_colors_params=None, where_secondary_overlap=None, run_reproject=True):

        if run_reproject:
            self.hdus = ReprojectCutouts(prime_cuts_hdu, secondary_cuts_hdu, where_secondary_overlap=where_secondary_overlap).hdus_repr
        else:
            self.hdus = stack_hdus(prime_cuts_hdu, secondary_cuts_hdu, where_secondary_overlap)


        if secondary_cuts_hdu is not None:
            secondary_cuts_hdu = np.atleast_1d(secondary_cuts_hdu)
            self.color_params = [prime_color_params, *secondary_cuts_hdu]
        else:
            self.color_params = [prime_color_params]

        #updates to last colors asked to be output from `get_col_obj` and therefore 'get_colored_arrays
        self.color_params_recently_used = self.color_params
        self.all_params_used = [self.color_params_recently_used]

        #self.col_objs = self.get_col_obj()

    @classmethod
    def from_filelist(cls):
        pass

    @classmethod
    def from_FitsCutter(cls, FitsCutter_obj, prime_color_params, secondary_colors_params=None, run_reproject=True):

        prime_hdus, secondaries_hdus = [FitsCutter_obj.get_cutout_hdus()]
        where_seconary_overlaps = FitsCutter_obj.where_secondary_overlap

        cls(prime_hdus, prime_color_params, secondaries_hdus, secondary_colors_params, where_secondary_overlap, run_reproject)

    @classmethod
    def from_full_files(cls, primary_filename, gridding, prime_color_params, secondary_filenames=None, hdu_index_primary=0, hdu_indices_secondaries=None, grid_method=half_step_overlap_grid, secondary_colors_params=None, run_reproject=True):

        return ColoredCutouts.from_FitsCutter(FitsCutter.from_fits(primary_filename, gridding, secondary_filenames, hdu_index_primary, hdu_indices_secondaries, grid_method), prime_color_params, secondary_colors_params, run_reproject)

    def get_colored_arrays(self, color_param_list=None, color=None, colortype='hex'):
        """
        This is a slow function

        Parameters
        ----------
        color_param_list : list of dictionaries, optional
            The parameters that affect the appearance of the colorized cutouts.
            The default is None.
        color : list of str or list of arraylike, optional
            Color of the colorized images. The default is None.
        colortype : str, optional
            If using a more specific colorization, need to tell the code
                whether `color` entered is hex or rgb. If a named color is
                given, this parameter doesn't matter. The default is 'hex'.

        Returns
        -------
        final_images : list of rgbarray
            List containing the final combined colorised cutouts.

        """
        color_objs = self.get_col_obj(color_param_list)
        colored = []

        for i in range(len(color_objs)):
            if color_objs[i].colored is None and color is not None:
                colored.append(color_objs[i].get_color(color[i], colortype))

            else:
                colored.append(color_objs[i].colored)

        colored_shuffled = []

        for i in range(len(colored[0])):
            colored_shuffled_column = []
            for j in range(len(color_objs)):
                if np.all(np.isnan(colored[j][i])):
                    continue #might need to append zeros for color balence sinstead of skipping?
                colored_shuffled_column.append(colored[j][i])
            colored_shuffled.append(colored_shuffled_column)

        colored = colored_shuffled
        del colored_shuffled

        final_images = [mcf.combine_multicolor(color, gamma=2.2) for color in colored]
        return final_images

    def get_stacked_images(self):
        """Gets the 2d arrays from the hdus then stacks them rowwise
        into a 3d cube
        """
        data3d = [get_img_from_hdu_list(hdu) for hdu in self.hdus]
        for i in range(len(data3d)):
            data3d[i] = np.array(data3d[i])

        return data3d


    def get_col_obj(self, color_param_list=None):
        """Wrapper to inialised  `ColorImages` for each
        color
        """
        color_objs = []
        if color_param_list is None:
            color_param_list = self.color_params_recently_used

        else:
            self.all_params_used.append(color_param_list)
            self.color_params_recently_used = color_param_list

        for i in range(np.shape(self.hdus)[1]):

            data2d = self.get_img_from_hdu_list(self.hdus[:,i])

            param = color_param_list[i]

            color_objs.append(ColorImages(data2d, rescalefn=param['rescalefn'], min_max=param['min_max'], color=param['color'], scaletype=param['scaletype'], colortype=param['colortype']))

        return color_objs


    def get_img_from_hdu_list(self, hdus):
        return [hdu.data for hdu in hdus]

    def save_colored_cutout_as_imagefile(self, format='png'):
        #Moved to FitsCutter
        pass


class CutoutObjects(object):
    """This class creates cutouts of a fits file according to a grid method.
    The default is a 50% overlap grid. Currently no user option to change
    the percentage
    """

    @u.quantity_input(gridding=u.arcmin)
    def __init__(self, hdu, gridding, grid_method=half_step_overlap_grid, remove_only_nans=False):
        self.nan_inds = []
        self.remove_only_nans = remove_only_nans
        self.hdu = hdu
        self.gridding = gridding
        self.grid_method = grid_method
        self.cutout_centers_pixel, self.gridding_odd_pixel = self.get_centers(return_pixel_gridding=True)
        if self.remove_only_nans:
            self.nan_inds = self._nan_inds()
            self.cutout_centers_pixel = self._remove_only_nans(self.cutout_centers_pixel)


        pixelsize = get_pixelsize_hdu(self.hdu)

        self.gridding_odd_arcmin = self.gridding_odd_pixel * pixelsize * 60 * u.arcmin

    @classmethod
    def from_fits(cls, fits_filename, gridding, hdu_index=0, grid_method=half_step_overlap_grid):

        # with fits.open(fits_filename) as hdulist:
        #     hdu = hdulist[hdu_index]
        hdu = fits.open(fits_filename)[hdu_index]

        return cls(hdu=hdu, gridding=gridding, grid_method=grid_method)

    def _remove_only_nans(self, param):

        param = np.delete(param, self.nan_inds, axis=0)
        return param

    def _nan_inds(self):
        percs   = self.percentage_of_nans()
        nan_inds = np.where(percs>50)[0]
        #self.nan_inds = nan_inds #only way for composition later
        return nan_inds

    def percentage_of_nans(self, cutout_objs=None, cutout_centers=None, gridding_odd=None):
        if cutout_objs is None:
            cutout_objs = self.get_cutouts(cutout_centers, gridding_odd)
        percs = np.zeros(len(cutout_objs))

        for i in range(len(cutout_objs)):
            image = cutout_objs[i].data
            percs[i] = get_perc_nan(image)

        return percs

    def get_centers(self, grid_method=None, return_pixel_gridding=False):
        """Method finds the xy indicies for tile/cutout centers


        Parameters
        ----------
        grid_method : TYPE, optional
            DESCRIPTION. The default is None.
        return_pixel_gridding : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        if grid_method is None:
            grid_method = self.grid_method

        header = self.hdu.header

        gridding_xy_pixel_odd  = self._get_lens(header, self.gridding)

        x_indices, y_indices = grid_method((header['NAXIS2'], header['NAXIS1']), gridding_xy_pixel_odd)

        xx, yy = np.meshgrid(x_indices, y_indices)
        centers_xy = np.column_stack([xx.ravel(), yy.ravel()])
        self._remove_only_nans(centers_xy)

        if return_pixel_gridding:
            return centers_xy, gridding_xy_pixel_odd.value
        else:
            return centers_xy


    def _get_lens(self, header, gridding):
        pixel_size = get_pixelsize_header(header) * 60 * u.arcmin

        xygrid = gridding/pixel_size

        xygrid = np.round(xygrid)

        #turns grid odd. Odd is better for gridding
        if xygrid[0] % 2 !=0:
            xygrid[0] +=1
        if xygrid[1] % 2 !=0:
            xygrid[1] +=1

        return xygrid

    #Not sure if catching will use up too much memory. Also seems quick so not needed?
    #@functools.lru_cache
    def get_cutouts(self, cutout_centers=None, gridding_odd=None, cutout_centers_ra_dec=None):

        if cutout_centers is None:
            cutout_centers = self.cutout_centers_pixel

        if cutout_centers_ra_dec is not None:
            cutout_centers = self.get_xy_centers_from_ra_dec(cutout_centers_ra_dec)


        if gridding_odd is None:
            gridding_odd = self.gridding_odd_pixel

        cutout_objs = []
        for cent in cutout_centers:
            try:
                cutout_objs.append(Cutout2D(self.hdu.data, position=cent,
                                          size=gridding_odd,
                                          wcs=wcs.WCS(self.hdu.header),
                                          mode='partial'))
            except NoOverlapError:
            # pass
                cutout_objs.append(None)


        return cutout_objs

    def get_centers_ra_dec(self, cutout_centers=None, hdu=None):
        if cutout_centers is None:
            cutout_centers = self.cutout_centers_pixel
        if hdu is None:
            hdu = self.hdu
        return convert_pixel2world(cutout_centers, hdu.header)

    def get_xy_centers_from_ra_dec(self, cutout_centers_ra_dec):
        return convert_world2pixel(cutout_centers_ra_dec, self.hdu.header)


    def get_cutout_properties(self, cutout_centers_ra_dec=None):

        if cutout_centers_ra_dec is not None:

            xy_cutout_centers = self.get_xy_centers_from_ra_dec(cutout_centers_ra_dec)
        else:
            xy_cutout_centers = self.cutout_centers_pixel
            cutout_centers_ra_dec = self.get_centers_ra_dec()

        perc_nan_in_cutout = self.percentage_of_nans()

        return xy_cutout_centers, cutout_centers_ra_dec, perc_nan_in_cutout


    def get_cutouts_footprint_ds9_region(self, ra_dec_cutout_centers=None):
        if ra_dec_cutout_centers is None:
            ra_dec_cutout_centers = self.get_centers_ra_dec()

        box_regions = [get_box_ds9_region(ra_dec, self.gridding_odd_arcmin) for ra_dec in ra_dec_cutout_centers]

        return box_regions


    def get_cutout_hdus(self, cutout_centers_ra_dec=None):
        cutouts_obj = self.get_cutouts(cutout_centers_ra_dec=cutout_centers_ra_dec)


        hdu_cutouts = []
        for i in range(len(cutouts_obj)):
            data, header = get_cutout_data_header(cutouts_obj[i])
            hdu_cutouts.append(update_hdu(copy.copy(self.hdu.copy()), data, header))

        return hdu_cutouts


    def save_fits_cutouts(self, path, band, ra_dec_cutout_centers=None):

        if ra_dec_cutout_centers is None:
            cutout_centers_ra_dec = self.get_centers_ra_dec()

        cutout_objs = self.get_cutouts(cutout_centers_ra_dec=cutout_centers_ra_dec)
        self._save(save_cutout_as_fits, cutout_objs, ra_dec_cutout_centers, path, band, hdu_obj_orig=copy.copy(self.hdu.copy()))

        return ra_dec_cutout_centers

    def save_as_ds9regions(self, path, ra_dec_cutout_centers=None):

        if ra_dec_cutout_centers is None:
            ra_dec_cutout_centers = self.get_centers_ra_dec()

        box_regions = self.get_cutouts_footprint_ds9_region(ra_dec_cutout_centers)
        savepath = os.path.join(path, 'cutout_regions_%.1f_%.1farcmin.reg' % tuple(self.gridding.value))
        Regions(box_regions).write(savepath, overwrite=True)

        return ra_dec_cutout_centers

    def _save(self, save_method, objs, ra_dec_cutout_centers, path, band=None, filename_method=zooniverse_cutout_filename_generator, **kwargs):

        if ra_dec_cutout_centers is None:
            ra_dec_cutout_centers = self.get_centers_ra_dec()

        for i in range(len(objs)):
            filename = filename_method(ra_dec_cutout_centers[i], self.gridding, band=band)

            savepath = os.path.join(path, filename)

            save_method(savepath, objs[i], **kwargs)


#Main class to interact with the program
class FitsCutter(object):
    """This function makes cutouts for a `primary`/main image and makes
    same sized and location cutouts of and additional(`secondary`) fits
    files given.
    Currently there might be some errors if a cutout being made
    on the secondary fitsfile is outside the range of the primary

    I added colorization as methods as composition to this class since it
    was convenient to have all the gridding params for saving. This makes
    the class a bit less focused, but means this is the only class you need
    to use to get the cutouts and save them.
    """

    # @u.quantity_input(gridding=u.arcmin)
    def __init__(self, prime_cuts, secondary_cuts=None):

        self.prime_cuts  = prime_cuts
        if not self.prime_cuts.remove_only_nans:
            self.prime_cuts = self._remove_nans(self.prime_cuts)

        self.gridding = self.prime_cuts.gridding_odd_arcmin
        self.cutout_centers_ra_dec_prime = self.prime_cuts.get_centers_ra_dec()

        if secondary_cuts is not None:
            self.secondary_cuts = np.atleast_1d(secondary_cuts)
            self.where_secondary_overlap = [np.where(np.array(sec_cut.get_cutouts(cutout_centers_ra_dec=self.cutout_centers_ra_dec_prime)) !=None)[0] for sec_cut in self.secondary_cuts]
            self.cutout_centers_ra_dec_secondaries = [self.cutout_centers_ra_dec_prime[inds] for inds in self.where_secondary_overlap]
        else:
            self.secondary_cuts = None

    @classmethod
    @u.quantity_input(gridding=u.arcmin)
    def from_hdu(cls, primary_hdu, gridding, secondary_hdus=None, grid_method=half_step_overlap_grid):

        gridding = gridding.to(u.arcmin)
        prime  = CutoutObjects(primary_hdu, gridding)
        if secondary_hdus is not None:
            secondary_hdus = np.atleast_1d(secondary_hdus)
            secs  = [CutoutObjects(hdu, gridding) for hdu in secondary_hdus]
        else:
            secs = None

        return cls(prime_cuts=prime, secondary_cuts=secs)


    @classmethod
    @u.quantity_input(gridding=u.arcmin)
    def from_fits(cls, primary_filename, gridding, secondary_filenames=None, hdu_index_primary=0,
                    hdu_indices_secondaries=None, grid_method=half_step_overlap_grid):

        gridding = gridding.to(u.arcmin)
        prime = CutoutObjects.from_fits(primary_filename, gridding, hdu_index_primary, grid_method=grid_method)
        if secondary_filenames is not None:
            secondary_filenames = np.atleast_1d(secondary_filenames)
            num = len(secondary_filenames)
            if hdu_indices_secondaries is None:
                hdu_indices_secondaries = [0] * num
            hdu_indices_secondaries = np.atleast_1d(hdu_indices_secondaries)
            secs  = [CutoutObjects.from_fits(secondary_filenames[i], gridding, hdu_indices_secondaries[i], grid_method=grid_method) for i in range(num)]
        else:
            secs = None

        return cls(prime_cuts=prime, secondary_cuts=secs)

    def _remove_nans(self, cut_obj):
        cut_obj.remove_only_nans = True
        cut_obj.nan_inds = cut_obj._nan_inds()
        cut_obj.cutout_centers_pixel = cut_obj._remove_only_nans(cut_obj.cutout_centers_pixel)

        return cut_obj


    def tabulate_cutout_properties(self, path, bands):
        # this should be the database linking function to keep track of
        # save names (ids?) parameters
        props = self.cutout_properties()

        pass

    def get_cutout_properties(self):

        xy_cutout_centers_prime, ra_dec_cutout_centers_prime, perc_nan_in_cutout_prime = self.prime_cuts.get_cutout_properties()

        if self.secondary_cuts is not None:
            all_sec_props = [sec_cut.get_cutout_properties(ra_dec_cutout_centers_prime) for sec_cut in self.secondary_cuts]
            xy_cutout_centers_secondaries, __, perc_nan_in_cutout_secondaries = np.column_stack(all_sec_props)
        else:
            xy_cutout_centers_secondaries = None
            perc_nan_in_cutout_secondaries = None

        return ra_dec_cutout_centers_prime, xy_cutout_centers_prime, perc_nan_in_cutout_prime, xy_cutout_centers_secondaries, perc_nan_in_cutout_secondaries

    def get_ds9_regions(self, ra_dec_cutout_centers=None):

        box_regions = self.prime_cuts.get_cutouts_footprint_ds9_region()

        return box_regions

    def get_cutouts(self):
        cutouts_prime = self.prime_cuts.get_cutouts()

        if self.secondary_cuts is not None:
            cutouts_secondaries = [self.secondary_cuts[i].get_cutouts(cutout_centers_ra_dec=self.cutout_centers_ra_dec_secondaries[i]) for i in range(len(self.secondary_cuts))]
        else:
            cutouts_secondaries = None

        return cutouts_prime, cutouts_secondaries

    def get_cutout_hdus(self):

        cutouts_prime, cutouts_secondaries = self.get_cutouts()

        hdu_primaries = self.prime_cuts.get_cutout_hdus()
        if self.secondary_cuts is not None:
            hdu_secondaries = [self.secondary_cuts[i].get_cutout_hdus(self.cutout_centers_ra_dec_secondaries[i]) for i in range(len(self.secondary_cuts))]
        else:
            hdu_secondaries = None


        return hdu_primaries, hdu_secondaries


    def get_ColoredCutouts(self, prime_color_params, secondary_colors_params=None):
        try:
            self.colored_obj
        except AttributeError:
            #------------adding a self so that we do not need to rerun
            #reprojection.
            prime_cuts_hdu, secondary_cuts_hdu = self.get_cutout_hdus()
            self.colored_obj = ColoredCutouts(prime_cuts_hdu, prime_color_params=prime_color_params, secondary_cuts_hdu=secondary_cuts_hdu, secondary_colors_params=secondary_colors_params, where_secondary_overlap=self.where_secondary_overlap)
            #prime_cuts_hdu, prime_color_params, secondary_cuts_hdu=None

        if secondary_colors_params is not None:
            params = [prime_color_params, *np.atleast_1d(secondary_colors_params)]
        else:
            params = [prime_color_params]
        colored_cutouts_rgb_arrays = self.colored_obj.get_colored_arrays(params)

        return colored_cutouts_rgb_arrays


    def save_final_rgbs_as_np_array(self, path, bands, colored_cutouts_rgb_arrays=None, filename_method=zooniverse_cutout_filename_generator, **kwargs):

        self._save_rgb(path, bands, np.save, colored_cutouts_rgb_arrays, filename_method, **kwargs)


    def save_colored_cutout_as_imagefile(self, path, bands, colored_cutouts_rgb_arrays=None, filename_method=zooniverse_cutout_filename_generator, format='png', **kwargs):
        format = kwargs.pop('format', format)
        origin = kwargs.pop('origin', 'lower')

        self._save_rgb(path=path, bands=bands, save_method=save_image, colored_cutouts_rgb_arrays=colored_cutouts_rgb_arrays, filename_method=filename_method, format=format, origin=origin, **kwargs)

    def _save_rgb(self, path, bands, save_method, colored_cutouts_rgb_arrays=None, filename_method=zooniverse_cutout_filename_generator, **kwargs):

        if colored_cutouts_rgb_arrays is None:
            #params from most recent usage are used
            colored_cutouts_rgb_arrays = self.colored_obj.get_colored_arrays()
        band = ''
        for i in range(len(bands)):
            band += '_' + bands[i]
        band = 'rgb_array' + band
        self.prime_cuts._save(save_method, colored_cutouts_rgb_arrays, None, path, band=band, filename_method=filename_method, **kwargs)

    def save_reprojected_cutout_as_fitscube(self, path, bands, colored_cutouts_rgb_arrays=None, filename_method=zooniverse_cutout_filename_generator):
        try:
            self.colored_obj
        except AttributeError:
            #------------adding a self so that we do not need to rerun
            #reprojection.
            prime_cuts_hdu, secondary_cuts_hdu = self.get_cutout_hdus()
            self.colored_obj = ColoredCutouts(prime_cuts_hdu, prime_color_params=prime_color_params, secondary_cuts_hdu=secondary_cuts_hdu, secondary_colors_params=secondary_colors_params, where_secondary_overlap=self.where_secondary_overlap)

        data3d_cube = self.colored_obj.get_stacked_images()

        cube_hdus = self.colored_obj.hdus[:,0]
        for i in range(len(self.colored_obj.hdus[:,0])):
            header_3d = make_header_3d(cube_hdus[i].header)

            cube_hdus[i] = update_hdu(cube_hdus[i], data3d_cube[i], header_3d)

        for i in range(len(bands)):
            band += '_' + bands[i]
        band = 'cube' + band
        self.prime_cuts._save(save_fitsfile, cube_hdus, path, band=band, filename_method=filename_method)


    def save_fits_cutouts(self, path, bands):


        bands = np.atleast_1d(bands)

        self.prime_cuts.save_fits_cutouts(path, bands[0])

        for i in range(len(bands)-1):
            self.secondary_cuts.save_fits_cutouts(path, bands[i+1], self.cutout_centers_ra_dec_secondaries[i])

    def save_as_ds9regions(self, path):
        self.prime_cuts.save_as_ds9regions(path)




WAVELENGTH = 6562.79
G = 6.67408e-11 #m^3/kg/s^2
PC = 3.0857e16 #m

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['mathtext.default'] = 'regular'

GALAXY='NGC0628'

DRIVE_PATH = 'C:\\Users\\Liz_J\\Documents\\'
JWST_PATH = DRIVE_PATH + GALAXY + '\\JWST\\'


jwst_770_filename = 'background_sub\\' + GALAXY.lower() + '_miri_f770w_anchored.fits' #'_miri_lv3_f770w_i2d_align.fits'
jwst_770_filepath = JWST_PATH + jwst_770_filename

miri_770, miri_770_header = get_fits_file_info(jwst_770_filepath)

#These are older version for me butam using them for the test
jwst_2100_filename = JWST_PATH + 'background_sub\\' + GALAXY.lower() + '_miri_lv3_f2100w_i2d_align.fits'
jwst_335_filename  = JWST_PATH + 'background_sub\\' + GALAXY.lower() + '_nircam_lv3_f335m_i2d_align.fits'


#testing initialisation with JWST images. WORKS!
test = FitsCutter.from_fits(jwst_770_filepath, gridding=[1,1] *u.arcmin, secondary_filenames=[jwst_2100_filename, jwst_335_filename], hdu_indices_secondaries=[1,1])

#testing that the cutouts look correct using region files: WORKS!
test.save_as_ds9regions(JWST_PATH)


#Colorisation params. `rescalefn` is the stretch (i.e., linear, log, sqrt etc.)
#`min_max` is the contast. `scaletype=abs' means the `min_max` are units of
# image. If `scaletype=perc' percentiles are used (between 0 and 100).
# `color` is the color. Can be a simple string of the primary addative and subtractive colors
# otherwise, needs to be a hex string or rgb list. If hex string, set `colortype` to `hex`
# or `rgb` for rgb list
miri_770_color = {
    'rescalefn':'asinh',
    'min_max':[0.1,10],
    'color':'red',
    'scaletype':'abs',
    'colortype':'hex'
}

miri_2100_color = {
    'rescalefn':'linear',
    'min_max':[-0.9,3],
    'color':'green',
    'scaletype':'abs',
    'colortype':'hex'
}

miri_335_color = {
    'rescalefn':'asinh',
    'min_max':[-0.2,3.2],
    'color':'cyan',
    'scaletype':'abs',
    'colortype':'hex'
}

#This gets the final colorized images using the above parameters. I
#have written it so that reprojection is only done once (though reprojection
#was fairly fast. Colorizing the data is slow. Do not know how to speed up
#as it uses a downloaded package). To run with different contrasts, change
#`min_max`

#IT WORKS: EACH COL IN COLS IS A COMBINED RGB IMAGE CREATED WITH THE ABOVE PARAMETERS
cols = test.get_ColoredCutouts(miri_770_color, [miri_2100_color, miri_335_color])

#testing image saving as pngs. If cols are not entered, the last run
#colorization is saved
#might want to made the autosave name include what colors/params were used
#in the generation of the image

test.save_colored_cutout_as_imagefile(
    path=JWST_PATH + 'cutout_test_images\\',
    bands=['F770W', 'F2100W', 'F335W'],
    colored_cutouts_rgb_arrays=cols
)


if __name__ == "__main__":
    pass



