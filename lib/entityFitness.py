# This script stores fitness functions for scoring the grid elements
# of a grid of scientific entities.

import yaml
from math import ceil

def initEntitybook (tWeight, entities, weights, colorcodes, geotiffs, weightDensity,
                    weightSpeed, weightAcc, defaultMargin, defaultRoam):

    # The 'entityBook' is a data storage for the analyst.
    # The analyst may reference this when determining
    # "interestingness"
    # Guide:
    # 'entities' : A list of N entities, where the index is ID, value is str name.
    #     EX: Entity '1' has name "spartina" ---> entities[1] == "spartina"
    # 'weights' : A list of N weights of the interest level of each entity.
    #     EX: Entity '1' has weight '1' ---> weights[1] == 1

    # To deal with 0-based counting, the '0'th entity is 'No entity'.
    #     Has weight of 0.

    entityBook = { 'tWeight'       : 0,
                   'entities'      : [],
                   'weights'       : [],
                   'weightDensity' : weightDensity,
                   'weightSpeed'   : weightSpeed,
                   'weightAcc'     : weightAcc,
                   'colorcodes'    : {},
                   'geotiffs'      : [],
                   'defaultMargin' : defaultMargin,
                   'defaultRoam'   : defaultRoam,
    }

    entityBook['tWeight'] = tWeight

    # Set 0th entry as 'No entity' with weight == 0
    entityBook['entities'].append("NULL")
    entityBook['weights'].append(0)
    entityBook['geotiffs'].append(None)

    entityBook['colorcodes'] = colorcodes
    entityBook['colorcodes'][255] = 0

    for e in entities:
        entityBook['entities'].append(e)
    for w in weights:
        entityBook['weights'].append(w)
    for g in geotiffs:
        entityBook['geotiffs'].append(g)

    return entityBook

def initEntitybookByFile (entityBookFile):

    # If no file provided, return None
    if entityBookFile == "None":
        entityBook = None
        return entityBook

    with open(entityBookFile) as f:
        config = yaml.safe_load(f)

    tWeight       = config['entityBook']['tWeight']
    entities      = config['entityBook']['entities']
    weights       = config['entityBook']['weights']
    colorcodes    = config['entityBook']['colorcodes']
    geotiffs      = config['entityBook']['geotiffs']
    weightDensity = config['entityBook']['weightDensity']
    weightSpeed   = config['entityBook']['weightSpeed']
    weightAcc     = config['entityBook']['weightAcc']
    defaultMargin = config['entityBook']['targetDefaults']['margin']
    defaultRoam   = config['entityBook']['targetDefaults']['roam']

    # Initialize with file-loaded arguments
    entityBook = initEntitybook (tWeight, entities, weights, colorcodes, geotiffs,
            weightDensity, weightSpeed, weightAcc, defaultMargin, defaultRoam)

    return entityBook

def entityBook2html (entityBook):

    # Generate html code with formatted presentation of
    # information stored in entityBook

    from yattag import Doc
    import json
    import pandas as pd

    doc, tag, text, = Doc ().tagtext ()

    # Non-tabular
    with tag ('p'):
        text ('Global weight of target influence: ' + str (entityBook['tWeight']))

    # Tabular
    colorcodes = [entityBook['colorcodes'].keys()[entityBook['colorcodes'].values().index(e)] for e in range (0,len(entityBook['entities']))]
    entities_tbl = pd.DataFrame (
        { 'NAME': entityBook['entities'],
          'WEIGHT': entityBook['weights'],
          'COLOR_CODE': colorcodes,
	})

    entities_tbl.columns.name = 'ENTITY'
    entities_html = entities_tbl.to_html ()
    doc.asis (entities_html)

    return doc.getvalue ()


def calcCellReward (grid, row, col, entityBook, fitfun = "rewardEntity"):

    # NOTE: Currently, applies a single fitness function using 'fitfun' param.
    # But what if a list of fitness funs to apply sequentially was provided?

    #####################
    # Fitness Functions #
    #####################

    def dummy ():
        # Cell fitness function: 'dummy'.
        # For testing. Always returns 0.
        return 0

    def rewardEntity (e, entityBook):
        # Cell fitness function: 'rewardEntity'
        # Cell's reward is based on the cell's entity value.
        # Looks up the weight of the entity found (if any) in entityBook.
        r = entityBook['weights'][e]
        return r

    def rewardSystem(e, densities, speeds, accs, entityBook):
        # Instead of giving reward based only the properties of a single cell,
        # uses measurements of a group of related cells.
        # Intuition: If a region of entity A is exhibiting unusual behavior,
        # then want to study each cell in that region.
        # Cell have reward based on their surrounding context.
        r = rewardEntity(e, entityBook)
        for ei in range(1, len(entityBook["entities"])):
            r = r + abs(densities[ei - 1]) * entityBook["weightDensity"] \
                    * entityBook["weights"][ei]
            r = r + abs(speeds   [ei - 1]) * entityBook["weightSpeed"] \
                    * entityBook["weights"][ei]
            r = r + abs(accs     [ei - 1]) * entityBook["weightAcc"] \
                    * entityBook["weights"][ei]
	return r

    # Init reward to 1
    r = 0
    # Get cell value
    c = grid[row][col]
    # convert cell value to entity
    e = entityBook['colorcodes'][c]

    #
    densities = [0 for e in range(len(entityBook["entities"]))]
    speeds    = [0 for e in range(len(entityBook["entities"]))]
    accs      = [0 for e in range(len(entityBook["entities"]))]

    for ei in range(1, len(entityBook["entities"])):
        densities[ei] = entityBook["grids"][ei - 1]["density"][row][col]
        speeds[ei]    = entityBook["grids"][ei - 1]["speed"][row][col]
        accs[ei]      = entityBook["grids"][ei - 1]["acc"][row][col]

    # Apply selected fitness function
    if fitfun == "dummy":
        r = dummy ()
    elif fitfun == "rewardEntity":
        r = rewardEntity (e, entityBook)
    elif fitfun == "rewardSystem":
        r = rewardSystem(e, densities, speeds, accs, entityBook)

    return r


def getRewardAlongLine (x0, y0, x1, y1, grid, xLimit, yLimit, entityBook):
    from bresenham import bresenham

    reward = 0
    x0 = int (ceil (x0))
    y0 = int (ceil (y0))
    x1 = int (ceil (x1))
    y1 = int (ceil (y1))

    # Get list of cells that are intersected by line segment
    b = list(bresenham(x0, y0, x1, y1))
    # Iterate over those cells
    for p in b:
        reward = reward + calcCellReward (grid, p[0], p[1], entityBook,
                                          fitfun = "rewardSystem")

    return reward



def rewardLine (path, xStart, yStart, xStop, yStop,
    xLimit, yLimit, gridTargets, entityBook):

    reward = { 'total' : 0.0, 'segments' : [] }

    if gridTargets is None:
        return reward

    xi = path[0]
    yi = path[0]
    xj = None
    yj = None

    # First segment is between start location and first waypoint
    reward['segments'].append (getRewardAlongLine (xStart, yStart, xi, yi,
                               gridTargets, xLimit, yLimit, entityBook))

    # Loop over interior waypoint segments
    count = 2
    iterations = len (path) / 2 - 1
    for i in range (0, iterations):
        xj = path[count]
        count = count + 1
        yj = path[count]
        count = count + 1

        reward['segments'].append (getRewardAlongLine (xi, yi, xj, yj, gridTargets,
                                   xLimit, yLimit, entityBook))

        xi = xj
        yi = yj

    # Final segment is between last waypoint and goal location
    reward['segments'].append (getRewardAlongLine (xj, yj, xStop, yStop,
                               gridTargets, xLimit, yLimit, entityBook))

    reward['total'] = sum (reward['segments'])

    return reward



