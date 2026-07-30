from netCDF4 import Dataset
import os
import numpy as np

# Format type: 'S1' or 'c' (NC_CHAR), 'i1' or 'b' or 'B' (NC_BYTE), 'u1' (NC_UBYTE),
# 'i2' or 'h' or 's' (NC_SHORT), 'u2' (NC_USHORT), 'i4' or 'i' or 'l' (NC_INT), 'u4' (NC_UINT),
# 'i8' (NC_INT64), 'u8' (NC_UINT64), 'f4' or 'f' (NC_FLOAT), 'f8' or 'd' (NC_DOUBLE)


class NetCDFVariable():
    def __init__(self, key, data):

        # Remove unecessary dimension (,1)
        if data.shape[-1] == 1:
            data = np.squeeze(data)
            
        self.key = key
        self.data = data
        self.format = data.dtype
        self.fillvalue = None
        self.units = None
        self.long_name = None
        self.dimensions = None


def write_netcdf(filename, dims, vars):
    """
    Create a netCDF file.
    
    :param filename: filename
    :param dims: list of dimensions (NetCDFVariable object)
    :param vars: list of variables (NetCDFVariable object)
    """
    
    # Remove file if already exist
    if os.path.isfile(filename):
        os.remove(filename)
    
    # Create netCDF file
    ncfile = Dataset(filename, mode='w', format='NETCDF4')
    
    # Create dimensions
    for dim in dims:
        if dim.data.ndim != 1:
            raise Exception(f"Error: {dim.data.key} dimension ndim equal {dim.data.ndim}, should be 1.\n")
        ncfile.createDimension(dim.key, dim.data.size)
    
    # Create variables
    for dim in dims:
        current_var = ncfile.createVariable(dim.key, dim.format, (dim.key,), fill_value=dim.fillvalue)
        # current_var._FillValue = dim.fillvalue
        current_var.units = dim.units
        current_var.long_name = dim.long_name
        current_var[:] = dim.data
        
    for var in vars:
        current_var = ncfile.createVariable(var.key, var.format, var.dimensions, fill_value=var.fillvalue)
        current_var.units = var.units
        current_var.long_name = var.long_name
        etendue = tuple([slice(var.data.shape[i]) for i in range(var.data.ndim)])
        current_var[etendue] = var.data
        
    # Close file
    ncfile.close()
    print(f"{filename} created.")
    
    return


if __name__ == '__main__':
    filename = 'test.nc'
    
    lat = NetCDFVariable('lat', np.arange(-89, 89.1, 2))
    lat.units = 'degrees_north'
    lat.long_name = 'latitude'
    lat.format = 'i2'
    
    lon = NetCDFVariable('lon', np.arange(1, 359.1, 2))
    lon.units = 'degrees_east'
    lon.long_name = 'longitude'
    lon.format = 'i2'
    
    time = NetCDFVariable('time', np.arange(1, 4))
    time.units = 'days'
    time.long_name = 'time'
    time.format = 'i2'
    
    dims = [lat, lon, time]
    
    map_1 = NetCDFVariable('map_1', np.ma.arange(lat.data.size*lon.data.size).reshape(lat.data.size, lon.data.size)/1000.)
    map_1.fillvalue = -9999.
    map_1.units = 'K'
    map_1.long_name = 'Skin temperature'
    map_1.dimensions = ('lat', 'lon')
    map_1.data[40:60, 50:140] = np.ma.masked
    map_1.format = 'f4'
    
    map_2 = NetCDFVariable('map_2', np.arange(lat.data.size*lon.data.size*time.data.size).reshape(lat.data.size, lon.data.size, time.data.size)/10.)
    map_2.fillvalue = -9999.
    map_2.units = 'K'
    map_2.long_name = 'Sea Surface Temperature'
    map_2.dimensions = ('lat', 'lon', 'time')
    map_2.format = 'f4'

    vars = [map_1, map_2]
    
    write_netcdf(filename, dims, vars)