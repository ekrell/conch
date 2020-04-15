# conch
Path planning for autonomous vehicles subject to environmental forces. 

Main features
- Handles discrete time-varying forces, such as water currents
- Handles reward in the space for balanced reward maximization and cost minimization
- Separation of library (utility) scrips for building planners and path planning scripts
- Focuses on path planning for unmanned surface vehicles


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


### Add libraries to Pythonpath
        MY_PATH=<your path to the conch repo>
        export PYTHONPATH=$PYTHONPATH:$MY_PATH/lib






