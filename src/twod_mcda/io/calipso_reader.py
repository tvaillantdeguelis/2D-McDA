import sys
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from readers.hdf_reader import HDF4Reader
from paths import CALIOP_DATA_HEAD_PATH, CALIOP_DATA_TAIL_PATH_FMT, \
    IIR_DATA_HEAD_PATH, IIR_DATA_TAIL_PATH_FMT, get_caliop_data_tail_path
from calipso_constants import *
from geotools import get_prof_min_max_indexes_from_lon
from standard_outputs import print_time
from calipso_calculator import compute_par_ab532, compute_ab_mol_and_b_mol, \
    nsf_from_V_domain_to_betap_domain, rms_from_P_domain_to_betap_domain, compute_shotnoise, \
    compute_backgroundnoise, make_molecular_model

MESSAGE_EXECPTION_SLICE_START_END_TYPE = '''f"Error: slice_start_end_type = '{slice_start_end_type}'" \
                                          "is not defined. Please use 'profindex' or 'longitude'\n"'''

class CALIPSOReader():
    def __init__(self, filepath):
        """
        Load parameters of a CALIPSO file.
        
        :param filepath: folderpath+filemane of file to load
        """
        self._data = {} # dict: (data, fillvalue)

        with HDF4Reader(filepath) as data_reader:
            # Load data with fillvalue
            sds_keys = data_reader.get_sds_keys()
            for sds_key in sds_keys:
                fill_value = data_reader.get_fillvalue(sds_key)
                if fill_value is None:
                    try:
                        # Get "fillvalue" attribute of CALIPSO variables (instead of "_FillValue")
                        fill_value = data_reader._sd_interface.select(sds_key).fillvalue
                    except AttributeError:
                        fill_value = FILL_VALUE_FLOAT # default (hopefully it will match)
                self._data[sds_key] = (data_reader.get_data(sds_key), fill_value)
            # Load metadata with fillvalue
            self._metadata_keys = data_reader.get_metadata_keys()
            for metadata_key in self._metadata_keys:
                fill_value = None
                self._data[metadata_key] = (data_reader.get_metadata(metadata_key), fill_value)
            # Get number of profiles
            self.nb_profiles = self._data['Latitude'][0].shape[0]
            
    def get_cal_keys(self):
        return self._data.keys()
    
    def get_fillvalue(self, key):
        return self._data[key][1]
        
    def get_data(self, key, slice_start=None, slice_end=None, slice_start_end_type='profindex',
                 do_fillvalue=True):
        """
        Get data for the key parameter from slice_start to slice_end.
        
        :param key: a CALIPSO parameter
        :param slice_start: (optional) start profile of the slice to load
                            default: the first profile
        :param slice_end: (optional) end profile of the slice to load (included)
                          default: the end of the data
        :param slice_start_end_type: 'profindex' if profile indexes provided or 'longitude' if
                                     longitudes provided (longitudes because increases/decreases
                                     monotonously on one granule unlike latitudes)
                                     default: 'profindex'
        :param do_fillvalue: mask where fillvalue
        :return: (masked) data array
        """
        
        if key in self._data.keys():
            data = self._data[key][0]
            fillvalue = self._data[key][1]
            if do_fillvalue:
                returned_data = np.ma.masked_where(data == fillvalue, data)
            else:
                returned_data = data
            if returned_data.shape[0] == self.nb_profiles:
                # Get slice start and end
                if slice_start_end_type == 'profindex':
                    prof_min, prof_max = slice_start, slice_end
                elif slice_start_end_type == 'longitude':
                    prof_min, prof_max = get_prof_min_max_indexes_from_lon(self._data['Longitude'][0],
                                                                           slice_start, slice_end)
                else:
                    raise Exception(MESSAGE_EXECPTION_SLICE_START_END_TYPE)
                if prof_max is not None:
                    prof_max_included = prof_max+1
                else:
                    prof_max_included = None
                # Slice selected section
                returned_data = returned_data[prof_min:prof_max_included] # works for any ndim (first is profiles)
            
            elif len(returned_data.shape) > 1 and returned_data.shape[1] == self.nb_profiles: # for 2D-McDA *_steps variables, profile dim is the second (=1)
                # Get slice start and end
                if slice_start_end_type == 'profindex':
                    prof_min, prof_max = slice_start, slice_end
                elif slice_start_end_type == 'longitude':
                    prof_min, prof_max = get_prof_min_max_indexes_from_lon(self._data['Longitude'][0],
                                                                           slice_start, slice_end)
                else:
                    raise Exception(MESSAGE_EXECPTION_SLICE_START_END_TYPE)
                if prof_max is not None:
                    prof_max_included = prof_max+1
                else:
                    prof_max_included = None
                # Slice selected section
                returned_data = returned_data[:, prof_min:prof_max_included, :] # 2D-McDA *_steps variables dims are (step, prof, alt)
        else:
            raise Exception(f"Error: key = '{key}' not found.\n")

        return returned_data


class CALIOPReader():
    def __init__(self, product, version, data_type, granule_date, slice_start=None, slice_end=None,
                 slice_start_end_type='profindex', folderpath=None):
        """
        Get original and derived CALIOP parameters on a regular grid.
        
        :param product: CALIOP data product ('L1', 'L2_VFM', ...)
        :param version: CALIOP version product (ex: 'V4.10')
        :param data_type: CALIOP data type (ex: 'Standard')
        :param granule_date: 'YYYY-MM-DDThh-mm-ssZx'
        :param folderpath: (optional) data path, if not given then try automatic path detection
        :param slice_start: (optional) start profile of the slice to load
                            default: the first profile
        :param slice_end: (optional) end profile of the slice to load (included)
                          default: the end of the data
        :param slice_start_end_type: 'profindex' if profile indexes provided or 'longitude' if
                                     longitudes provided (longitudes because increases/decreases
                                     monotonously on one granule unlike latitudes)
                                     default: 'profindex'
        """
        self.product = product
        self.version = version
        self.data_type = data_type
        self.granule_date = granule_date
        if folderpath:
            self.folderpath = folderpath
        else: # automatic path detection
            self.folderpath = automatic_path_detection(product, version, data_type, granule_date)
        
        # Get filepath
        self.filename = CAL_LID_FILENAME_FMT % (product, data_type, version.replace(".", "-"),
                                                granule_date)
        self.filepath = os.path.join(self.folderpath, self.filename)
        
        # Load file
        self.data_reader = CALIPSOReader(self.filepath)
        
        # Get prof_min and prof_max
        lon = self.data_reader.get_data('Longitude')
        lat = self.data_reader.get_data('Latitude')
        if lon.ndim != 1:
            if lon.shape[1] == 3:
                lon = lon[:, 1]
                lat = lat[:, 1]
            else:
                raise Exception(f"TO DO")
        if slice_start_end_type == 'profindex':
            self.prof_min, self.prof_max = slice_start, slice_end
        elif slice_start_end_type == 'longitude':
            self.prof_min, self.prof_max = get_prof_min_max_indexes_from_lon(lon, slice_start, slice_end)
        else:
            raise Exception(f"Error: slice_start_end_type = '{slice_start_end_type}' is not defined. "
                     "Please use 'profindex' or 'longitude'\n")
        self.lon_min = lon[self.prof_min]
        self.lat_min = lat[self.prof_min]
        self.lon_max = lon[self.prof_max]
        self.lat_max = lat[self.prof_max]
        self.nb_profiles = self.prof_max - self.prof_min + 1
        

    def get_data(self, key, do_fillvalue=True):
        """
        Get data on regular grid.
        
        :param key: a CALIPSO parameter
        :param do_fillvalue: mask where fillvalue
        :return: (masked) data array
        """
        return self.data_reader.get_data(key, self.prof_min, self.prof_max, 'profindex', do_fillvalue)

        
class CALIOPRegularGridReader():
    def __init__(self, product, version, data_type, granule_date, grid='333mx30m',
                 slice_start=None, slice_end=None, slice_start_end_type='profindex',
                 index30m_alt_max=None, folderpath=None, version_l1=None, data_type_l1=None,
                 deconvolution=False):
        """
        Get original and derived CALIOP parameters on a regular grid.
        
        :param product: CALIOP data product ('L1', 'L2_VFM', ...)
        :param version: CALIOP version product (ex: 'V4.10')
        :param data_type: CALIOP data type (ex: 'Standard')
        :param granule_date: 'YYYY-MM-DDThh-mm-ssZx'
        :param grid: (optional) regular grid on which to put the data: '333m×30m', '1kmx60m', '5kmx60m', or '5kmx180m'
                     default: '333mx30m' (CALIOP full resolution)
        :param folderpath: (optional) data path, if not given then try automatic path detection
        :param slice_start: (optional) start profile of the slice to load (at 333 m resolution)
                            default: the first profile
        :param slice_end: (optional) end profile of the slice to load (included) (at 333 m resolution)
                          default: the end of the data
        :param slice_start_end_type: 'profindex' if profile indexes provided or 'longitude' if
                                     longitudes provided (longitudes because increases/decreases
                                     monotonously on one granule unlike latitudes)
                                     default: 'profindex'
        """
        self.product = product
        self.version = version
        self.data_type = data_type
        self.hgrid = grid.split('x')[0]
        self.vgrid = grid.split('x')[1]
        self.granule_date = granule_date
        self.index30m_alt_max = index30m_alt_max
        self.filename = CAL_LID_FILENAME_FMT % (product, data_type, version.replace(".", "-"),
                                                granule_date)
        if folderpath:
            self.folderpath = folderpath
        else: # automatic path detection
            self.folderpath = automatic_path_detection(product, version, data_type, granule_date)
        self.filepath = os.path.join(self.folderpath, self.filename)
        self.deconvolution = deconvolution
        self._molecular_profiles = {"532": None, "532par": None, "532per": None, "1064": None}
        
        self.data_reader = CALIPSOReader(self.filepath)
        
        # Get lat and lon on the whole granule
        self.lon_granule = np.copy(self.data_reader.get_data('Longitude'))
        self.lat_granule = np.copy(self.data_reader.get_data('Latitude'))
        
        # Get prof_min and prof_max from 333 m resolution lat/lon
        if self.product not in PRODUCT_H_RESOLUTION['333m']:
            try:
                self._lon_granule_l1 = self.data_reader.get_data('ssLongitude')
                self._lat_granule_l1 = self.data_reader.get_data('ssLatitude')
            except:
                if (version_l1 is None) or (data_type_l1 is None):
                    raise Exception("No latitude/longitude at 333 m resolution in the file. Please "
                                    "specify version_l1 and data_type_l1 when generating the "
                                    "CALIOPRegularGridReader object.")
                # Load lat/lon from corresponding L1 file
                cal_l1 = CALIOPRegularGridReader(product='L1',
                                                 version=version_l1,
                                                 data_type=data_type_l1,
                                                 granule_date=self.granule_date,
                                                 slice_start=slice_start,
                                                 slice_end=slice_end,
                                                 slice_start_end_type=slice_start_end_type)
                self._lon_granule_l1 = cal_l1.data_reader.get_data('Longitude')
                self._lat_granule_l1 = cal_l1.data_reader.get_data('Latitude')
        else:
            self._lon_granule_l1 = np.copy(self.lon_granule)
            self._lat_granule_l1 = np.copy(self.lat_granule)

        if slice_start_end_type=='profindex':
            if slice_start:
                if slice_start >= 0:
                    self.prof_min = int(slice_start)
                elif slice_start < 0:
                    self.prof_min = self._lon_granule_l1.size + int(slice_start)
            else:
                self.prof_min = 0
            if slice_end:
                self.prof_max = int(slice_end)
            else:
                self.prof_max =  self._lat_granule_l1.size - 1
        elif slice_start_end_type=='longitude':
            self.prof_min, self.prof_max = get_prof_min_max_indexes_from_lon(self._lon_granule_l1, slice_start,
                                                                             slice_end)
        else:
            raise Exception(MESSAGE_EXECPTION_SLICE_START_END_TYPE)
        self.lon_min = self._lon_granule_l1[self.prof_min]
        self.lat_min = self._lat_granule_l1[self.prof_min]
        self.lon_max = self._lon_granule_l1[self.prof_max]
        self.lat_max = self._lat_granule_l1[self.prof_max]
        self.nb_profiles = self.prof_max - self.prof_min + 1

    def print_lat_lon_min_max(self):
        print(f"\tFrom min profile index {self.prof_min:d} "
              f"(lat = {self.lat_min:.2f} / "
              f"lon = {self.lon_min:.2f}) "
              f"to max profile index {self.prof_max:d} "
              f"(lat = {self.lat_max:.2f} / "
              f"lon = {self.lon_max:.2f})")
    
    def get_data(self, key, do_fillvalue=True):
        """
        Get data on regular grid.
        
        :param key: a CALIPSO parameter
        :param do_fillvalue: mask where fillvalue
        :return: (masked) data array
        """
        if (key == 'Latitude') and (self.product not in PRODUCT_H_RESOLUTION['333m']):
            data = np.copy(self._lat_granule_l1)
        elif (key == 'Longitude') and (self.product not in PRODUCT_H_RESOLUTION['333m']):
            data = np.copy(self._lon_granule_l1)
        elif key in self.data_reader.get_cal_keys():
            data = self.data_reader.get_data(key, None, None, 'profindex', do_fillvalue)
            # Check if variable of profiles
            if data.shape[0] == self.lat_granule.shape[0]:
                var_of_profiles = True
            else:
                var_of_profiles = False
            if self.product in PRODUCT_H_RESOLUTION['333m']:
                pass
            elif self.product in PRODUCT_H_RESOLUTION['1km']:
                if (data.ndim >= 1) and var_of_profiles:
                    data = duplicate_1km_to_333m(data)
            elif (self.product in PRODUCT_H_RESOLUTION['5km']) or (self.product == "L2_VFM"):
                if key == "Feature_Classification_Flags":
                    data = unfold_vfm(data, put_in_all_alt_grid=True)
                elif key[:2] != "ss":
                    if (data.ndim >= 1) and var_of_profiles:
                        data = duplicate_5km_to_333m(data)
            else:
                raise Exception("Error: Product unknown or horizontal resolution of product unknown.")
            
            # Add zeros when nb profiles is different from L1 (at the end of the granule)
            if var_of_profiles:
                nb_missing_prof = self._lat_granule_l1.shape[0] - data.shape[0] # doesn't work for VFM because _lat_granule_l1 is taken from VFM SS variables
                if nb_missing_prof > 0:
                    data = add_zeros_where_missing_profiles(data, nb_missing_prof)
                
            # Slice selected section
            if var_of_profiles:
                data = data[self.prof_min:self.prof_max+1] # works for any ndim (first is profiles)

            # If layer variable then convert to grid
            try:
                key_is_layer_var = ((self.product == 'L2_05kmALay') or
                                    (self.product == 'L2_05kmCLay') or
                                    (self.product == 'L2_05kmMLay')) and \
                                   (self.data_reader.get_data(key).shape[1] == N_LAYERS_05KM_LAYER_PRODUCT) or \
                                   ((self.product == 'L2_01kmCLay')) and \
                                   (self.data_reader.get_data(key).shape[1] == N_LAYERS_01KM_LAYER_PRODUCT) or \
                                   ((self.product == 'L2_333mMLay')) and \
                                   (self.data_reader.get_data(key).shape[1] == N_LAYERS_333M_LAYER_PRODUCT)
            except:
                key_is_layer_var = False
            if key_is_layer_var:
                alt_base = self.data_reader.get_data("Layer_Base_Altitude", None, None, 'profindex', do_fillvalue)
                alt_top = self.data_reader.get_data("Layer_Top_Altitude", None, None, 'profindex', do_fillvalue)
                if self.product in PRODUCT_H_RESOLUTION['333m']:
                    alt_base = alt_base[self.prof_min:self.prof_max+1, :]
                    alt_top = alt_top[self.prof_min:self.prof_max+1, :]
                    data = self.layer2grid(data, alt_base, alt_top)
                elif self.product in PRODUCT_H_RESOLUTION['1km']:
                    alt_base = duplicate_1km_to_333m(alt_base)
                    alt_top = duplicate_1km_to_333m(alt_top)
                    alt_base = alt_base[self.prof_min:self.prof_max+1, :]
                    alt_top = alt_top[self.prof_min:self.prof_max+1, :]
                    data = self.layer2grid(data, alt_base, alt_top)
                elif self.product in PRODUCT_H_RESOLUTION['5km']:
                    alt_base = duplicate_5km_to_333m(alt_base)
                    alt_top = duplicate_5km_to_333m(alt_top)
                    alt_base = alt_base[self.prof_min:self.prof_max+1, :]
                    alt_top = alt_top[self.prof_min:self.prof_max+1, :]
                    data = self.layer2grid(data, alt_base, alt_top)
            
        elif key == "Lidar_Data_Altitudes_init":
            data = self.data_reader.get_data("Lidar_Data_Altitudes", self.prof_min, self.prof_max,
                                             'profindex', do_fillvalue)
        
        elif key == "Parallel_Attenuated_Backscatter_532":
            data = self._get_par_ab532(do_fillvalue)
        
        elif key == "Molecular_Total_Attenuated_Backscatter_532":
            data, _ = self._get_molecular_profiles(532, '', do_fillvalue)
        elif key == "Molecular_Parallel_Attenuated_Backscatter_532":
            data, _ = self._get_molecular_profiles(532, 'par', do_fillvalue)
        elif key == "Molecular_Perpendicular_Attenuated_Backscatter_532":
            data, _ = self._get_molecular_profiles(532, 'per', do_fillvalue)
        elif key == "Molecular_Attenuated_Backscatter_1064":
            data, _ = self._get_molecular_profiles(1064, '', do_fillvalue)
         
        elif key == "Molecular_Total_Backscatter_532":
            _, data = self._get_molecular_profiles(532, '', do_fillvalue)
        elif key == "Molecular_Parallel_Backscatter_532":
            _, data = self._get_molecular_profiles(532, 'par', do_fillvalue)
        elif key == "Molecular_Perpendicular_Backscatter_532":
            _, data = self._get_molecular_profiles(532, 'per', do_fillvalue)
        elif key == "Molecular_Backscatter_1064":
            _, data = self._get_molecular_profiles(1064, '', do_fillvalue)
            
        #### Compute NSF in beta' domain (NSF in level 1B data is in V = P / Ga domain)
        elif key == "Noise_Scale_Factor_532_Parallel_AB_domain":
            data = self._get_nsf_in_ab_domain(532, 'par', do_fillvalue)
        elif key == "Noise_Scale_Factor_532_Perpendicular_AB_domain":
            data = self._get_nsf_in_ab_domain(532, 'per', do_fillvalue)
        elif key == "Noise_Scale_Factor_1064_AB_domain":
            data = self._get_nsf_in_ab_domain(1064, '', do_fillvalue)
            
        #### Compute RMS in beta' domain (RMS in level 1B data is in P domain)
        elif key == "Parallel_RMS_Baseline_532_AB_domain":
            data = self._get_rms_in_ab_domain(532, 'par', do_fillvalue)
        elif key == "Perpendicular_RMS_Baseline_532_AB_domain":
            data = self._get_rms_in_ab_domain(532, 'per', do_fillvalue)
        elif key == "RMS_Baseline_1064_AB_domain":
            data = self._get_rms_in_ab_domain(1064, '', do_fillvalue)
            
        #### Compute range-dependent uncertainty based on molecular model
        elif key == "Shot_Noise_532_Parallel":
            data = self._get_shotnoise(532, 'par', do_fillvalue)
        elif key == "Shot_Noise_532_Perpendicular":
            data = self._get_shotnoise(532, 'per', do_fillvalue)
        elif key == "Shot_Noise_1064":
            data = self._get_shotnoise(1064, '', do_fillvalue)

        #### Compute range-independent uncertainty from background RMS ####
        elif key == "Background_Noise_532_Parallel":
            data = self._get_backgroundnoise(532, 'par', do_fillvalue)
        elif key == "Background_Noise_532_Perpendicular":
            data = self._get_backgroundnoise(532, 'per', do_fillvalue)
        elif key == "Background_Noise_1064":
            data = self._get_backgroundnoise(1064, '', do_fillvalue)
        
        #### Compute scattering ratio uncertainty standard deviation ####
        elif key == "Attenuated_Scattering_Ratio_Uncertainty_Standard_Deviation_532_Parallel":
            shotnoise_532_par = self._get_shotnoise(532, 'par', do_fillvalue)
            bkgnoise_532_par = self._get_backgroundnoise(532, 'par', do_fillvalue)
            data = np.sqrt(shotnoise_532_par**2 + bkgnoise_532_par**2)
        elif key == "Attenuated_Scattering_Ratio_Uncertainty_Standard_Deviation_532_Perpendicular":
            shotnoise_532_per = self._get_shotnoise(532, 'per', do_fillvalue)
            bkgnoise_532_per = self._get_backgroundnoise(532, 'per', do_fillvalue)
            data = np.sqrt(shotnoise_532_per**2 + bkgnoise_532_per**2)
        elif key == "Attenuated_Scattering_Ratio_Uncertainty_Standard_Deviation_1064":
            shotnoise_1064 = self._get_shotnoise(1064, '', do_fillvalue)
            bkgnoise_1064 = self._get_backgroundnoise(1064, '', do_fillvalue)
            data = np.sqrt(shotnoise_1064**2 + bkgnoise_1064**2)
        else:
            raise Exception(f"Error: unknown key = {key}.\n")

        # Put on regular grid (new resolution but grid stays 300m×30m) 
        if key == "Lidar_Data_Altitudes":
            if data.size == NUMBER_OF_VERTICAL_BINS:
                data = alt_to_regular_30m_vertical_grid(data)
                data = data[:self.index30m_alt_max]
            # elif data.size == NUMBER_OF_VERTICAL_BINS_05KMPRO:
            #     data = alt05kmpro_to_regular_30m_vertical_grid(data)
        elif data.ndim == 1:
            # No vertical averaging for 1D data
            pass
        elif data.ndim == 2:
            if data.shape[1] == NUMBER_OF_VERTICAL_BINS:
                data = shape_to_regular_30m_vertical_grid(data)
                if self.deconvolution:
                    if key == "Parallel_Attenuated_Backscatter_532":
                        npzfile = np.load(os.path.split(os.path.join(os.path.dirname(__file__)))[0] +
                                          '/deconvolution_matrices.npz')
                        inv_mat_par = npzfile['inv_mat_par']
                        for i in np.arange(data.shape[0]):
                            data[i, :] = np.matmul(inv_mat_par, data[i, ::-1].filled(0))[::-1]
                            # [::-1] because profile top to bottom <=> bottom to top / fill mask values with 0 which should work with the deconvolution
                    elif key == "Perpendicular_Attenuated_Backscatter_532":
                        npzfile = np.load(os.path.split(os.path.join(os.path.dirname(__file__)))[0] +
                                          '/deconvolution_matrices.npz')
                        inv_mat_per = npzfile['inv_mat_per']
                        for i in np.arange(data.shape[0]):
                            data[i, :] = np.matmul(inv_mat_per, data[i, ::-1].filled(0))[::-1]
                            # [::-1] because profile top to bottom <=> bottom to top / fill mask values with 0 which should work with the deconvolution
                    elif key == "Total_Attenuated_Backscatter_532": # sum deconvoluted par and per
                        par_ab_532 = self.get_data("Parallel_Attenuated_Backscatter_532", do_fillvalue)
                        per_ab_532 = self.get_data("Perpendicular_Attenuated_Backscatter_532", do_fillvalue)
                        data = par_ab_532 + per_ab_532
                data = data[:, :self.index30m_alt_max]
                data = from_30mx333m_to_new_resolution(data, self.vgrid, self.hgrid, self.prof_min)
            elif data.shape[1] == NUMBER_OF_VERTICAL_BINS_MET:
                if key == "Temperature":
                    data = self._interp_temperature(data, do_fillvalue)
                    data = shape_to_regular_30m_vertical_grid(data)
                    data = data[:, :self.index30m_alt_max]
                    data = from_30mx333m_to_new_resolution(data, self.vgrid, self.hgrid, self.prof_min)
        else:
            raise Exception(f"Error: key = {key} with ndim ≥ 3 not implemented yet.\n")
        

        return data
    
    def layer2grid(self, data, alt_base, alt_top):
        """Replace layer properties in a grid using base and top altitudes"""
        data_grid = np.ma.ones((alt_base.shape[0], NUMBER_OF_VERTICAL_BINS))*FILL_VALUE_FLOAT
        for i_prof in np.arange(alt_base.shape[0]):
            for i_layer in np.arange(alt_base.shape[1]):
                if not alt_top.mask[i_prof, i_layer]: # if not masked (i.e. if layer exists)
                    alt_top_i = np.abs(alt_top[i_prof, i_layer] - LIDAR_DATA_ALTITUDES).argmin()
                    alt_base_i = np.abs(alt_base[i_prof, i_layer] - LIDAR_DATA_ALTITUDES).argmin()
                    data_grid[i_prof, alt_top_i:alt_base_i+1] = data[i_prof, i_layer]
        data_grid = np.ma.masked_where(data_grid == FILL_VALUE_FLOAT, data_grid)
        return data_grid
    
    def _get_par_ab532(self, do_fillvalue):
        tot_ab_532 = self.data_reader.get_data("Total_Attenuated_Backscatter_532", self.prof_min,
                                               self.prof_max, 'profindex', do_fillvalue)
        per_ab_532 = self.data_reader.get_data("Perpendicular_Attenuated_Backscatter_532",
                                                self.prof_min, self.prof_max, 'profindex',
                                                do_fillvalue)
        par_ab_532 = compute_par_ab532(tot_ab_532, per_ab_532)
        return par_ab_532
    
    def _get_molecular_profiles(self, wl, polar, do_fillvalue):
        mol_nd = self.data_reader.get_data("Molecular_Number_Density", self.prof_min,
                                           self.prof_max, 'profindex', do_fillvalue)
        O3_nd = self.data_reader.get_data("Ozone_Number_Density", self.prof_min,
                                          self.prof_max, 'profindex', do_fillvalue)
        alt = self.data_reader.get_data("Lidar_Data_Altitudes", self.prof_min,
                                        self.prof_max, 'profindex', do_fillvalue)
        met_alt = self.data_reader.get_data("Met_Data_Altitudes", self.prof_min,
                                            self.prof_max, 'profindex', do_fillvalue)
        if self._molecular_profiles[str(wl)+polar] is None:
            self._molecular_profiles[str(wl)+polar] = compute_ab_mol_and_b_mol(mol_nd, O3_nd, alt,
                                                                               met_alt, wl, polar)
        return self._molecular_profiles[str(wl)+polar]
    
    def _get_nsf_in_ab_domain(self, wl, polar, do_fillvalue):
        sat_alt = self.data_reader.get_data("Spacecraft_Altitude", self.prof_min,
                                            self.prof_max, 'profindex', do_fillvalue)[:, np.newaxis]
        caliop_lidar_tilt = self.data_reader.get_data("Off_Nadir_Angle", self.prof_min,
                                                      self.prof_max, 'profindex', do_fillvalue)[:, np.newaxis]
        data_alt = self.data_reader.get_data("Lidar_Data_Altitudes", self.prof_min,
                                             self.prof_max, 'profindex', do_fillvalue)[np.newaxis,:]
        range_alt = range_from_altitude(sat_alt, data_alt, caliop_lidar_tilt)
        pgr = np.array((1,))
        if wl == 532:
            calibration_cst = self.data_reader.get_data("Calibration_Constant_532", self.prof_min,
                                                        self.prof_max, 'profindex', do_fillvalue)[:, np.newaxis]
            laser_energy = self.data_reader.get_data("Laser_Energy_532", self.prof_min,
                                                     self.prof_max, 'profindex', do_fillvalue)[:, np.newaxis]
            if polar == 'par':
                nsf = self.data_reader.get_data("Noise_Scale_Factor_532_Parallel", self.prof_min,
                                                self.prof_max, 'profindex', do_fillvalue)[:, np.newaxis]
            if polar == 'per':
                nsf = self.data_reader.get_data("Noise_Scale_Factor_532_Perpendicular", self.prof_min,
                                                self.prof_max, 'profindex', do_fillvalue)[:, np.newaxis]
                pgr = self.data_reader.get_data("Depolarization_Gain_Ratio_532", self.prof_min,
                                                self.prof_max, 'profindex', do_fillvalue)[:, np.newaxis]
        elif wl == 1064:
            calibration_cst = self.data_reader.get_data("Calibration_Constant_1064", self.prof_min,
                                                        self.prof_max, 'profindex', do_fillvalue)[:, np.newaxis]
            laser_energy = self.data_reader.get_data("Laser_Energy_1064", self.prof_min,
                                                     self.prof_max, 'profindex', do_fillvalue)[:, np.newaxis]
            nsf = self.data_reader.get_data("Noise_Scale_Factor_1064", self.prof_min,
                                            self.prof_max, 'profindex', do_fillvalue)[:, np.newaxis]
        else:
            raise Exception(f"Error: Unrecognized wavelength: {wl}; use 532 or 1064 instead\n\n")
        return nsf_from_V_domain_to_betap_domain(nsf, range_alt, laser_energy, calibration_cst, pgr)
    
    def _get_rms_in_ab_domain(self, wl, polar, do_fillvalue):
        sat_alt = self.data_reader.get_data("Spacecraft_Altitude", self.prof_min,
                                            self.prof_max, 'profindex', do_fillvalue)[:, np.newaxis]
        caliop_lidar_tilt = self.data_reader.get_data("Off_Nadir_Angle", self.prof_min,
                                         self.prof_max, 'profindex', do_fillvalue)[:, np.newaxis]
        data_alt = self.data_reader.get_data("Lidar_Data_Altitudes", self.prof_min,
                                             self.prof_max, 'profindex', do_fillvalue)[np.newaxis,:]
        range_alt = range_from_altitude(sat_alt, data_alt, caliop_lidar_tilt)
        pgr = np.array((1,))
        if wl == 532:
            calibration_cst = self.data_reader.get_data("Calibration_Constant_532", self.prof_min,
                                                        self.prof_max, 'profindex', do_fillvalue)[:, np.newaxis]
            laser_energy = self.data_reader.get_data("Laser_Energy_532", self.prof_min,
                                                     self.prof_max, 'profindex', do_fillvalue)[:, np.newaxis]
            if polar == 'par':
                rms = self.data_reader.get_data("Parallel_RMS_Baseline_532", self.prof_min,
                                                self.prof_max, 'profindex', do_fillvalue)[:, np.newaxis]
                gain = self.data_reader.get_data("Parallel_Amplifier_Gain_532", self.prof_min,
                                                 self.prof_max, 'profindex', do_fillvalue)[:, np.newaxis]
            if polar == 'per':
                rms = self.data_reader.get_data("Perpendicular_RMS_Baseline_532", self.prof_min,
                                                self.prof_max, 'profindex', do_fillvalue)[:, np.newaxis]
                gain = self.data_reader.get_data("Perpendicular_Amplifier_Gain_532", self.prof_min,
                                                 self.prof_max, 'profindex', do_fillvalue)[:, np.newaxis]
                pgr = self.data_reader.get_data("Depolarization_Gain_Ratio_532", self.prof_min,
                                                self.prof_max, 'profindex', do_fillvalue)[:, np.newaxis]
        elif wl == 1064:
            calibration_cst = self.data_reader.get_data("Calibration_Constant_1064", self.prof_min,
                                                        self.prof_max, 'profindex', do_fillvalue)[:, np.newaxis]
            laser_energy = self.data_reader.get_data("Laser_Energy_1064", self.prof_min,
                                                     self.prof_max, 'profindex', do_fillvalue)[:, np.newaxis]
            rms = self.data_reader.get_data("RMS_Baseline_1064", self.prof_min,
                                            self.prof_max, 'profindex', do_fillvalue)[:, np.newaxis]
            gain = self.data_reader.get_data("Amplifier_Gain_1064", self.prof_min,
                                             self.prof_max, 'profindex', do_fillvalue)[:, np.newaxis]
        else:
            raise Exception(f"Error: Unrecognized wavelength: {wl}; use 532 or 1064 instead\n\n")
        return rms_from_P_domain_to_betap_domain(rms, range_alt, laser_energy, gain, calibration_cst, pgr)
    
    def _get_shotnoise(self, wl, polar, do_fillvalue):
        nb_bins_shift = self.data_reader.get_data("Number_Bins_Shift", self.prof_min,
                                                  self.prof_max, 'profindex', do_fillvalue)
        nb_bins_shift_abs = np.squeeze(np.abs(nb_bins_shift))
        nsf = self._get_nsf_in_ab_domain(wl, polar, do_fillvalue)
        mol_ab, _ = self._get_molecular_profiles(wl, polar, do_fillvalue)
        fcorr = get_caliop_correction_function(wl)
        nb_pixels = get_nb_pixels(wl)
        return compute_shotnoise(fcorr, nb_bins_shift_abs, nb_pixels, nsf, mol_ab)
    
    def _get_backgroundnoise(self, wl, polar, do_fillvalue):
        nb_bins_shift = self.data_reader.get_data("Number_Bins_Shift", self.prof_min,
                                                  self.prof_max, 'profindex', do_fillvalue)
        nb_bins_shift_abs = np.squeeze(np.abs(nb_bins_shift))
        rms = self._get_rms_in_ab_domain(wl, polar, do_fillvalue)
        mol_ab, _ = self._get_molecular_profiles(wl, polar, do_fillvalue)
        fcorr = get_caliop_correction_function(wl)
        nb_pixels = get_nb_pixels(wl)
        return compute_backgroundnoise(fcorr, nb_bins_shift_abs, nb_pixels, rms, mol_ab)
    
    def _interp_temperature(self, met_temp, do_fillvalue):
        met_alt = self.data_reader.get_data("Met_Data_Altitudes", self.prof_min,
                                            self.prof_max, 'profindex', do_fillvalue)
        alt = self.data_reader.get_data("Lidar_Data_Altitudes", self.prof_min, self.prof_max,
                                        'profindex', do_fillvalue).filled(FILL_VALUE_FLOAT)
        # Replace last filled value (subsurface) with surface value (last not filled value)
        last_notmasked_index = np.ma.notmasked_edges(met_temp, axis=1)[1]
        for i in range(last_notmasked_index[0].size):
            met_temp[i, last_notmasked_index[1][i] + 1 :] = met_temp[i, last_notmasked_index[1][i]]
        met_temp = met_temp.filled(0) # Put 0 where still filled value (hopefully not) because it can't be mask array for interp
        # Interpolate to get temperature values for all lidar data alt
        f = interp1d(met_alt, met_temp)
        temp = f(alt)
        return temp


def automatic_path_detection(product, version, data_type, granule_date):
    if CALIOP_DATA_HEAD_PATH:
        caliop_data_tail_path = get_caliop_data_tail_path(product, version, data_type, granule_date)
        return os.path.join(CALIOP_DATA_HEAD_PATH, caliop_data_tail_path)
    else:
        raise Exception("Error: Data paths are not defined for this machine. Please add data paths "
                 "to my_modules/readers/paths.py or give explicit folderpath when creating a "
                 "CALIOPRegularGridReader object instance.\n")
    

def alt_to_regular_30m_vertical_grid(alt, reverse_altitude=True):
    """
    Put alt in regular grid (30 m)
    reverse_altitude: if True, return array from bottom to top
    """

    # Initialization
    nb_vert_levels = get_nb_regular_30m_vertical_levels()
    reg_grid_alt = np.ones(nb_vert_levels)*FILL_VALUE_FLOAT
    reg_grid_r5_index_range = (0, N_BINS_R5*N_30M_BINS_PER_BIN_R5)
    reg_grid_r4_index_range = (reg_grid_r5_index_range[1],
                               reg_grid_r5_index_range[1] + N_BINS_R4*N_30M_BINS_PER_BIN_R4)
    reg_grid_r3_index_range = (reg_grid_r4_index_range[1],
                               reg_grid_r4_index_range[1] + N_BINS_R3*N_30M_BINS_PER_BIN_R3)
    reg_grid_r2_index_range = (reg_grid_r3_index_range[1],
                               reg_grid_r3_index_range[1] + N_BINS_R2*N_30M_BINS_PER_BIN_R2)
    reg_grid_r1_index_range = (reg_grid_r2_index_range[1],
                               reg_grid_r2_index_range[1] + N_BINS_R1*N_30M_BINS_PER_BIN_R1)
    
    # Copy region R2 (30 m resolution)
    reg_grid_alt[reg_grid_r2_index_range[0]:reg_grid_r2_index_range[1]] = \
        alt[LAYER_ALTITUDE_R2_INDEX_RANGE[0]:LAYER_ALTITUDE_R2_INDEX_RANGE[1]+1]
    step_30m = alt[LAYER_ALTITUDE_R2_INDEX_RANGE[1]-2] - alt[LAYER_ALTITUDE_R2_INDEX_RANGE[1]-1]
    
    # Complete R1 by decreasing by nb_bins×step_30m the lowest bin altitude of R2
    start_index = reg_grid_r1_index_range[0]
    end_index = reg_grid_r1_index_range[1]
    reg_grid_alt[start_index:end_index] = reg_grid_alt[start_index-1] - (np.arange(end_index-start_index)+1)*step_30m
    
    # Complete R3, R4, and R5 by increasing by nb_bins×step_30m the highest bin altitude of R2
    end_index = reg_grid_r3_index_range[1]
    reg_grid_alt[end_index-1::-1] = reg_grid_alt[end_index] + (np.arange(end_index)+1)*step_30m

    if reverse_altitude:
        reg_grid_alt = reg_grid_alt[::-1]

    return reg_grid_alt


def alt05kmpro_to_regular_30m_vertical_grid(alt, reverse_altitude=True):
    """
    Put alt in regular grid (30 m)
    reverse_altitude: if True, return array from bottom to top
    """

    

    if reverse_altitude:
        reg_grid_alt = reg_grid_alt[::-1]

    return reg_grid_alt


def shape_to_regular_30m_vertical_grid(data, reverse_altitude=True):
    """
    Duplicate and/or average CALIOP data to get a regular 30 m vertical resolution grid.
    reverse_altitude: if True, return array from bottom to top
    """
    
    # Initialization
    nb_vert_levels = get_nb_regular_30m_vertical_levels()
    reg_grid_r5_index_range = (0, N_BINS_R5*N_30M_BINS_PER_BIN_R5)
    reg_grid_r4_index_range = (reg_grid_r5_index_range[1],
                               reg_grid_r5_index_range[1] + N_BINS_R4*N_30M_BINS_PER_BIN_R4)
    reg_grid_r3_index_range = (reg_grid_r4_index_range[1],
                               reg_grid_r4_index_range[1] + N_BINS_R3*N_30M_BINS_PER_BIN_R3)
    reg_grid_r2_index_range = (reg_grid_r3_index_range[1],
                               reg_grid_r3_index_range[1] + N_BINS_R2*N_30M_BINS_PER_BIN_R2)
    reg_grid_r1_index_range = (reg_grid_r2_index_range[1],
                               reg_grid_r2_index_range[1] + N_BINS_R1*N_30M_BINS_PER_BIN_R1)

    reg_grid_data = np.ma.ones((data.shape[0], nb_vert_levels))*FILL_VALUE_FLOAT

    # Duplicate data
    reg_grid_data[:, reg_grid_r5_index_range[0]:reg_grid_r5_index_range[1]] = \
        np.repeat(data[:, LAYER_ALTITUDE_R5_INDEX_RANGE[0]:LAYER_ALTITUDE_R5_INDEX_RANGE[1]+1],
                  N_30M_BINS_PER_BIN_R5, axis=1)
    reg_grid_data[:, reg_grid_r4_index_range[0]:reg_grid_r4_index_range[1]] = \
        np.repeat(data[:, LAYER_ALTITUDE_R4_INDEX_RANGE[0]:LAYER_ALTITUDE_R4_INDEX_RANGE[1]+1],
                  N_30M_BINS_PER_BIN_R4, axis=1)
    reg_grid_data[:, reg_grid_r3_index_range[0]:reg_grid_r3_index_range[1]] = \
        np.repeat(data[:, LAYER_ALTITUDE_R3_INDEX_RANGE[0]:LAYER_ALTITUDE_R3_INDEX_RANGE[1]+1],
                  N_30M_BINS_PER_BIN_R3, axis=1)
    reg_grid_data[:, reg_grid_r2_index_range[0]:reg_grid_r2_index_range[1]] = \
        np.repeat(data[:, LAYER_ALTITUDE_R2_INDEX_RANGE[0]:LAYER_ALTITUDE_R2_INDEX_RANGE[1]+1],
                  N_30M_BINS_PER_BIN_R2, axis=1)
    reg_grid_data[:, reg_grid_r1_index_range[0]:reg_grid_r1_index_range[1]] = \
        np.repeat(data[:, LAYER_ALTITUDE_R1_INDEX_RANGE[0]:LAYER_ALTITUDE_R1_INDEX_RANGE[1]+1],
                  N_30M_BINS_PER_BIN_R1, axis=1)
    
    if reverse_altitude:
        reg_grid_data = reg_grid_data[:, ::-1]
    
    return reg_grid_data


def get_nb_regular_30m_vertical_levels():
    
    nb_30m_vert_levels = N_BINS_R1*N_30M_BINS_PER_BIN_R1 + \
                         N_BINS_R2*N_30M_BINS_PER_BIN_R2 + \
                         N_BINS_R3*N_30M_BINS_PER_BIN_R3 + \
                         N_BINS_R4*N_30M_BINS_PER_BIN_R4 + \
                         N_BINS_R5*N_30M_BINS_PER_BIN_R5
    
    return nb_30m_vert_levels


def unfold_vfm(vfm, put_in_all_alt_grid=False):
    """
    Unfold VFM in regular grid with 545 levels.
    
    :param vfm: folded VFM
    :param put_in_all_alt_grid: (optional) put in grid with 583 levels (-2 to 40 km). Zero where no data.
                                default: False
    :return: unfolded VFM
    """
    # Number of VFM masks in the file
    nb_vfm = vfm.shape[0]

    # Initialization
    vfm_unfolded = np.zeros((nb_vfm*N_LASER_PULSES_PER_5km, NUMBER_OF_VFM_VERTICAL_BINS),
                            dtype='uint16')
    r4_index_range = (0, 0+N_BINS_R4) # Regions R1 and R5 are not in VFM
    r3_index_range = (r4_index_range[1], r4_index_range[1]+N_BINS_R3)
    r2_index_range = (r3_index_range[1], r3_index_range[1]+N_BINS_R2)
    
    # Loop on each VFM mask in the file
    i_vfm = 0
    while i_vfm < nb_vfm:
        # 20.2 to 30.1 km
        for i in np.arange(N_PROFILES_VFM_R4):
            start = i_vfm*N_LASER_PULSES_PER_5km+i*N_333M_BINS_PER_BIN_R4
            vfm_unfolded[start:start+N_333M_BINS_PER_BIN_R4, r4_index_range[0]:r4_index_range[1]] = \
                vfm[i_vfm, i*N_BINS_R4:(i+1)*N_BINS_R4]
        # 8.2 to 20.2 km
        for i in np.arange(N_PROFILES_VFM_R3):
            start = i_vfm*N_LASER_PULSES_PER_5km+i*N_333M_BINS_PER_BIN_R3
            index_first_bin_r3 = N_PROFILES_VFM_R4*N_BINS_R4
            vfm_unfolded[start:start+N_333M_BINS_PER_BIN_R3, r3_index_range[0]:r3_index_range[1]] =\
                vfm[i_vfm, index_first_bin_r3+i*N_BINS_R3:index_first_bin_r3+(i+1)*N_BINS_R3]
        # -0.5 to 8.2 km
        for i in np.arange(N_PROFILES_VFM_R2):
            index_first_bin_r2 = N_PROFILES_VFM_R4*N_BINS_R4+N_PROFILES_VFM_R3*N_BINS_R3
            vfm_unfolded[i_vfm*N_LASER_PULSES_PER_5km+i, r2_index_range[0]:r2_index_range[1]] =\
                vfm[i_vfm, index_first_bin_r2+i*N_BINS_R2:index_first_bin_r2+(i+1)*N_BINS_R2]
        i_vfm += 1
    
    if put_in_all_alt_grid:
        vfm_unfolded_all = np.zeros((nb_vfm*N_LASER_PULSES_PER_5km, NUMBER_OF_VERTICAL_BINS),
                                    dtype='uint16')
        vfm_unfolded_all[:, LAYER_ALTITUDE_R4_INDEX_RANGE[0]:LAYER_ALTITUDE_R2_INDEX_RANGE[1]+1] = vfm_unfolded
        vfm_unfolded = vfm_unfolded_all
        
    return vfm_unfolded


def get_single_shot_index_from_5km_index(i_5km):
    """
    Return min and max single shot indexes of a 5 km profile index
    """
    ss_min = i_5km*N_LASER_PULSES_PER_5km
    ss_max = (i_5km+1)*N_LASER_PULSES_PER_5km - 1

    return ss_min, ss_max


def split_granule_date(granule_date):
    
    granule_date_dict = {}
    granule_date_dict['year'] = int(granule_date[:4])
    granule_date_dict['month'] = int(granule_date[5:7])
    granule_date_dict['day'] = int(granule_date[8:10])
    granule_date_dict['hour'] = int(granule_date[11:13])
    granule_date_dict['min'] = int(granule_date[14:16])
    granule_date_dict['sec'] = int(granule_date[17:19])
    daynight_flag = granule_date[19:21]
    if daynight_flag == 'ZD':
        granule_date_dict['daynigth'] = 'day'
    elif daynight_flag == 'ZN':
        granule_date_dict['daynigth'] = 'night'
    
    return granule_date_dict


def duplicate_5km_to_333m(data_5km):
    return np.ma.repeat(data_5km, 15, axis=0)


def duplicate_1km_to_333m(data_1km):
    return np.ma.repeat(data_1km, 3, axis=0)

def add_zeros_where_missing_profiles(data, nb_missing_prof):
    if data.ndim == 1:
        return np.ma.concatenate((data, np.ma.zeros(nb_missing_prof)))
    else:
        return np.ma.vstack((data, np.ma.zeros((nb_missing_prof, data.shape[1]))))

def from_30mx333m_to_new_resolution(data, vgrid, hgrid, prof_min, print_first_ID=False):
    """
    Average CALIOP data to a new resolution.
    Attention: the data resolution change but the grid stays 30mx333m.
    """
    
    # Look for first profile of a 5 km chunk
    if prof_min:
        profID_first_in_chunk = get_first_profileID_of_chunk(prof_min)
    else:
        profID_first_in_chunk = 0
    if print_first_ID:
        print("First profile of first 5 km chunk: %d" % profID_first_in_chunk)
    
    if vgrid == '30m':
            pass # already in 30 m resolution
    elif vgrid == '60m':
        # Average 30 m to 60 m
        data_copy = np.ma.copy(data)
        for i in range(0, data.shape[1], 2):
            for j in range(i, i+2):
                data[:, j] = np.ma.mean(data_copy[:, i:i+2], axis=1)
    elif vgrid == '180m':
        # Average 30 m to 180 m
        data_copy = np.ma.copy(data)
        # for i in np.arange(data_copy.shape[1]):
        #     print(f"{i:3d} {data_copy[20, i]:15.10f}")
        # stop
        for i in range(0, data.shape[1], 6)[:-1]:
            for j in range(i, i+6):
                data[:, j] = np.ma.mean(data_copy[:, i:i+6], axis=1)
    else:
        raise Exception('Error: vgrid unknown')
    
    if hgrid == '333m':
        pass # already at 333 m resolution
    elif hgrid == '1km':
        data_copy = np.ma.copy(data) # data already averaged vertically
        for i in range(profID_first_in_chunk, data.shape[0]-N_LASER_PULSES_PER_1km+1, N_LASER_PULSES_PER_1km):
            for j in range(i, i+N_LASER_PULSES_PER_1km):
                data[j, :] = np.ma.mean(data_copy[i:i+N_LASER_PULSES_PER_1km, :], axis=0)
    elif hgrid == '5km':
        data_copy = np.ma.copy(data) # data already averaged vertically
        for i in range(profID_first_in_chunk, data.shape[0]-N_LASER_PULSES_PER_5km+1, N_LASER_PULSES_PER_5km):
            for j in range(i, i+N_LASER_PULSES_PER_5km):
                data[j, :] = np.ma.mean(data_copy[i:i+N_LASER_PULSES_PER_5km, :], axis=0)
    else:
        raise Exception('Error: hgrid unknown')
    
    return data


def get_first_profileID_of_chunk(profID):
    """
    Look for first profile of a 5 km chunk
    :param profID: profile ID in a chunk
    :return: first profile ID of the chunk
    """
    
    modulo_prof = profID % N_LASER_PULSES_PER_5km
    profID_first_in_chunk = 15 - modulo_prof
    if profID_first_in_chunk == 15:
        profID_first_in_chunk = 0
    
    return profID_first_in_chunk


def range_from_altitude(spacecraft_alt, data_alt, caliop_lidar_tilt):
    """Get range distance between the satellite and the bin"""
    return (spacecraft_alt - data_alt)/np.cos(caliop_lidar_tilt*np.pi/180.)

    
if __name__ == '__main__':
    
    from datetime import datetime
    
    
    GRANULE_DATE = "2010-03-21T02-17-14ZN"
    VERSION_CAL_LID_L1 = "V4.10"
    TYPE_CAL_LID_L1 = "Standard"
    cal_l1 = CALIOPRegularGridReader(product='L1',
                                     version=VERSION_CAL_LID_L1,
                                     data_type=TYPE_CAL_LID_L1,
                                     granule_date=GRANULE_DATE)
    
    sds_keys = [
        "Shot_Noise_532_Parallel",
        "Shot_Noise_532_Perpendicular",
        "Shot_Noise_1064",
        "Background_Noise_532_Parallel",
        "Background_Noise_532_Perpendicular",
        "Background_Noise_1064",
        "Attenuated_Scattering_Ratio_Uncertainty_Standard_Deviation_532_Parallel",
        "Attenuated_Scattering_Ratio_Uncertainty_Standard_Deviation_532_Perpendicular",
        "Attenuated_Scattering_Ratio_Uncertainty_Standard_Deviation_1064"
    ]

    data_dict_l1 = {}
    for sds_key in sds_keys:
        print(f"Load {sds_key}")
        data_dict_l1[sds_key] = cal_l1.get_data(sds_key)

    print(data_dict_l1["Shot_Noise_532_Parallel"])
    print(data_dict_l1["Shot_Noise_532_Perpendicular"])
    print(data_dict_l1["Shot_Noise_1064"])
    print(data_dict_l1["Background_Noise_532_Parallel"])
    print(data_dict_l1["Background_Noise_532_Perpendicular"])
    print(data_dict_l1["Background_Noise_1064"])
    print(data_dict_l1["Attenuated_Scattering_Ratio_Uncertainty_Standard_Deviation_532_Parallel"])
    print(data_dict_l1["Attenuated_Scattering_Ratio_Uncertainty_Standard_Deviation_532_Perpendicular"])
    print(data_dict_l1["Attenuated_Scattering_Ratio_Uncertainty_Standard_Deviation_1064"])
    stop
    
    # ---------------------------
    # Test on CAL_LID_L2_VFM file
    filename_vfm = "/home/thibault/Documents/Travail/codes/DATA/CALIOP/VFM.v4.20/2010" \
                   "/2010_06_01/CAL_LID_L2_VFM-Standard-V4-20.2010-06-01T01-33-28ZN.hdf"

    vfm_min_index, vfm_max_index = 0, 2

    sds_keys = [
        "Profile_ID",
        "Feature_Classification_Flags",
        "Latitude",
        "Longitude",
        "ssLatitude",
        "Land_Water_Mask"
    ]

    ss_sds_keys = [
        "ssLatitude",
        "ssLongitude"
    ]

    metadata_keys = [
        "GEOS_Version",
        "Product_ID"
    ]

    calispo_reader = CALIPSOReader(filename_vfm)

    data_dict_vfm = {}
    for sds_key in sds_keys:
        data_dict_vfm[sds_key] = calispo_reader.get_data(sds_key, slice_start=vfm_min_index,
                                                         slice_end=vfm_max_index)

    ss_vfm_min_index = get_single_shot_index_from_5km_index(vfm_min_index)[0]
    ss_vfm_max_index = get_single_shot_index_from_5km_index(vfm_max_index)[1]
    ss_data_dict_vfm = {}
    for ss_sds_key in ss_sds_keys:
        ss_data_dict_vfm[ss_sds_key] = calispo_reader.get_data(ss_sds_key,
                                                               slice_start=ss_vfm_min_index,
                                                               slice_end=ss_vfm_max_index)

    metadata_dict_vfm = {}
    for metadata_key in metadata_keys:
        metadata_dict_vfm[metadata_key] = calispo_reader.get_data(metadata_key)

    if True:
        for key, values in data_dict_vfm.items():
            print(key, values)

    if True:
        print("Latitude:", data_dict_vfm["Latitude"])
        print("ssLatitude:", ss_data_dict_vfm["ssLatitude"])

    if True:
        for key, values in metadata_dict_vfm.items():
            print(key, values)


# -----------------------
    # Test on CAL_IIR_L2 file
    filename_iir_l2 = "/home/thibault/Documents/Travail/codes/DATA/IIR/CAL_IIR_L2.v4.20/"\
                      "2010/2010_06_01/CAL_IIR_L2_Track-Standard-V4-20.2010-06-01T01-33-28ZN.hdf"

    # with HDF4Reader(filename_iir_l2) as data_reader:
    #     sds_keys_got = data_reader.get_sds_keys()

    prof_min_index, prof_max_index = 0, 150

    sds_keys = [
        "Latitude",
        "Date_Time_at_Granule_End",
        "Mean_10_60_Brightness_Temp_All"
    ]

    calispo_reader = CALIPSOReader(filename_iir_l2)

    data_dict_l2 = {}
    for sds_key in sds_keys:
        data_dict_l2[sds_key] = calispo_reader.get_data(sds_key, slice_start=prof_min_index,
                                                        slice_end=prof_max_index)

    if False:
        for key, values in data_dict_l2.items():
            print(key, values)
            