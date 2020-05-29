# conch
Path planning for autonomous vehicles subject to environmental forces. 

**Undergoing complete code overhaul**

**Details/tutorial coming soon**


## Installation

### Dependencies

**PyGMO**
- Install dependencies
-- **boost**
        sudo apt-get install libboost-all-dev
-- **nlopt**
        sudo apt-get install libnlopt*
- Follow [these steps](https://esa.github.io/pygmo/install.html), but disable SNOPT and IPOPT
-- (Sept. 1, 2019) Had to apply [pull request #197](https://github.com/esa/pagmo/pull/197)
- Update library links, cache
        ldconfig

**netCDF4**
        sudo pip2 install netcdf4

**geopy**
        sudo pip2 install geopy

**bresenham**
        sudo pip2 install bresenham







