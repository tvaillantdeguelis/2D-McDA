"""Read CALIOP products on the regular grid expected by 2D-McDA."""

import os

import numpy as np
from twod_mcda.caliop.derived import CALIOPDerivedVariablesMixin
from twod_mcda.caliop.hdf import HDF4Reader
from twod_mcda.caliop.paths import automatic_path_detection
from twod_mcda.caliop.constants import (
    CAL_LID_FILENAME_FMT,
    FILL_VALUE_FLOAT,
    LIDAR_DATA_ALTITUDES,
    NUMBER_OF_VERTICAL_BINS,
    NUMBER_OF_VERTICAL_BINS_MET,
    N_LAYERS_01KM_LAYER_PRODUCT,
    N_LAYERS_05KM_LAYER_PRODUCT,
    N_LAYERS_333M_LAYER_PRODUCT,
    PRODUCT_H_RESOLUTION,
)
from twod_mcda.caliop.geography import get_prof_min_max_indexes_from_lon
from twod_mcda.caliop.grids import (
    add_zeros_where_missing_profiles,
    alt_to_regular_30m_vertical_grid,
    duplicate_1km_to_333m,
    duplicate_5km_to_333m,
    from_30mx333m_to_new_resolution,
    shape_to_regular_30m_vertical_grid,
    unfold_vfm,
)
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

        
class CALIOPRegularGridReader(CALIOPDerivedVariablesMixin):
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
        lidar_data_altitudes = LIDAR_DATA_ALTITUDES
        if lidar_data_altitudes is None:
            try:
                lidar_data_altitudes = self.data_reader.get_data(
                    "Lidar_Data_Altitudes"
                )
            except Exception as exc:
                raise RuntimeError(
                    "Lidar_Data_Altitudes is required to regrid layer data."
                ) from exc

        data_grid = np.ma.ones((alt_base.shape[0], NUMBER_OF_VERTICAL_BINS))*FILL_VALUE_FLOAT
        for i_prof in np.arange(alt_base.shape[0]):
            for i_layer in np.arange(alt_base.shape[1]):
                if not alt_top.mask[i_prof, i_layer]: # if not masked (i.e. if layer exists)
                    alt_top_i = np.abs(
                        alt_top[i_prof, i_layer] - lidar_data_altitudes
                    ).argmin()
                    alt_base_i = np.abs(
                        alt_base[i_prof, i_layer] - lidar_data_altitudes
                    ).argmin()
                    data_grid[i_prof, alt_top_i:alt_base_i+1] = data[i_prof, i_layer]
        data_grid = np.ma.masked_where(data_grid == FILL_VALUE_FLOAT, data_grid)
        return data_grid
