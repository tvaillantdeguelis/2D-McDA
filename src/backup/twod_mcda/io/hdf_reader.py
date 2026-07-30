from pyhdf.HDF import HDF
from pyhdf.SD import SD
from pyhdf.error import HDF4Error


class HDF4Reader:
    """
    Read HDF4 file using the "with" paradigm.
    """
    
    def __init__(self, file_path):
        self.file_path = file_path
        self._sd_interface = None
        self._hdf_interface = None
        self._vs_interface = None
        self._metadata = None
    
    # __enter__ method for the "with" statement
    def __enter__(self):
        try:
            self._sd_interface = SD(self.file_path)
            self._hdf_interface = HDF(self.file_path)
            self._vs_interface = self._hdf_interface.vstart()
        except HDF4Error:
            if self._vs_interface:
                self._vs_interface.end()
            if self._sd_interface:
                self._sd_interface.end()
            if self._hdf_interface:
                self._hdf_interface.close()
            print("Error: '%s' does not exist." % self.file_path)
            raise
        try:
            self._load_metadata()
        except HDF4Error:
            self._metadata = {}
        return self

    # __exit__ method for the "with" statement
    def __exit__(self, *args):
        if self._vs_interface:
            self._vs_interface.end()
        if self._sd_interface:
            self._sd_interface.end()
        if self._hdf_interface:
            self._hdf_interface.close()

    def _load_metadata(self):
        metadata = self._vs_interface.attach("metadata")
        field_infos = metadata.fieldinfo()
        all_data = metadata.read(metadata._nrecs)[0]
        metadata.detach()
        self._metadata = {}
        field_name_index = 0
        for field_info, data in zip(field_infos, all_data):
            self._metadata[field_info[field_name_index]] = data
        return self

    def get_metadata(self, key):
        return self._metadata[key]

    def get_metadata_keys(self):
        return self._metadata.keys()

    def is_key_in_metadata(self, key):
        """
        Check if parameter (key) is in metadata.
        """
        return key in self._metadata.keys()

    def get_sds_keys(self):
        """
        Get all of the top level SDSs.
        """
        return self._sd_interface.datasets()

    def get_data(self, key, do_squeeze=True):
        try:
            data = self._sd_interface.select(key).get()
        except HDF4Error:
            print("Error: '%s' is not in %s." % (key, self.file_path))
            raise

        if do_squeeze:
            data = data.squeeze()

        return data
    
    def get_fillvalue(self, key):
        try:
            return self._sd_interface.select(key).getfillvalue()
        except HDF4Error:
            return None

if __name__ == '__main__':

    filename_vfm = "/home/vaillant/DATA_CALIOP/VFM.v4.20/2010/2010_06_01/CAL_LID_L2_VFM-Standard-V4-20.2010-06-01T01-33-28ZN.hdf"
    filepath_l1 = "/home/vaillant/DATA_CALIOP/CAL_LID_L1.v4.10/2010/2010_03_21/CAL_LID_L1-Standard-V4-10.2010-03-21T02-17-14ZN.hdf"
    filepath_2dmcda = "/home/vaillant/codes/projects/2D_CALIOP/2D_McDA/out/2D_McDA.v1.01/2010/2010_06_06/CAL_LID_L2_2D_McDA-Prototype-V1-01.2010-06-06T00-13-08ZN_lon_31.00_29.00.hdf"

    with HDF4Reader(filepath_l1) as data_reader:
        sds_keys = data_reader.get_sds_keys()
    
    for sds_key in sds_keys:
        print(sds_key, sds_keys[sds_key])
