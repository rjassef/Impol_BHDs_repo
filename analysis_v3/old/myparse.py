import argparse

def myparse():

    parser = argparse.ArgumentParser(description='Estimate polarization')

    #Flags to redo some calculations.
    parser.add_argument('--use_skyflats', action='store_true', help='Use the data calibrated with sky flats.')
    parser.add_argument('--force_new', action='store_true', help='Force recalculation of the masks and cosmic ray cleaned images.')
    parser.add_argument('--show_plots', action='store_true', help='Show plots if any are made.')


    #Parse the command line arguments
    args = parser.parse_args()

    return args
