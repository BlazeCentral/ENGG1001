"""
ENGG1001 Assignment 2
Semester 1, 2023
"""

#Import libraries
from satdb import SatDB
import math
import numpy as np
import matplotlib.pyplot as plt


#Details
__author__ = "Blaise_Delforce"
__email__ = "blaise.delforce@outlook.cok"

""""
CODE
"""

#TASK 1  - Computing Eccentricity - DONE

def compute_eccentricity(apogee_alt, perigee_alt, radius):
    """Return eccentricity of orbit.

    Parameters
    ----------
    apogee_alt : float
        Altitude of satellite at apogee, units: km
    perigee_alt : float
        Altitude of satellite at perigee, units: km
    radius : float
        Radius of body (Earth) at focus of orbit, units: km
        
    Return
    ------
    eccentricity: float
        Eccentricity of orbit
    """

    eccentricity = (apogee_alt - perigee_alt)/(apogee_alt + perigee_alt + 2*radius)
    return eccentricity

    
#TASK 2 - Computing Period
def compute_semi_major_axis(apogee_alt, perigee_alt, radius):
    """Return semi major axis of orbit

    Paramaters
    ----------
    apogee_alt : float
        Altitude of satellite at apogee, units: km
    perigee_alt : float
        Altitude of satellite at perigee, units: km
    radius : float
        Radius of body (Earth) at focus of orbit, units: km

    Return
    ------
    semi_major_axis: float
        Semi Major Axis of Orbis
    """
    semi_major_axis = 0.5*(apogee_alt + perigee_alt + 2*radius)
    return semi_major_axis


def compute_period(apogee_alt, perigee_alt, radius, grav_param):
    """Return period of orbit.

    Parameters
    ----------
    apogee_alt : float
        Altitude of satellite at apogee, units: km
    perigee_alt : float
        Altitude of satellite at perigee, units: km
    radius : float
        Radius of body (Earth) at focus of orbit, units: km
    grav_param : float
        Gravitational parameter of body (Earth), units: km^3/s^2
        
    Return
    ------
    period_hours: float
        Period of orbit, units: hours
    """
    #Calculates semi-major axis, uses that to calculate period.
    semi_major_axis = compute_semi_major_axis(apogee_alt, perigee_alt, radius)
    period_seconds = (2*math.pi*(semi_major_axis ** (3/2)))/math.sqrt(grav_param)
    period_hours = period_seconds/3600
    return period_hours
    
#TASK 3 - Plot Orbital Path 
def plot_orbit_path(apogee_alt, perigee_alt, radius, n_samples):
    """Produce plot of orbit path.

    Parameters
    ----------
    apogee_alt : float
        Altitude of satellite at apogee, units: km
    perigee_alt : float
        Altitude of satellite at perigee, units: km
    radius : float
        Radius of body (Earth) at focus of orbit, units: km
    n_samples : int
        Number of samples along orbit path for use in plot
    """
    #Calculates relevant variables
    e = compute_eccentricity(apogee_alt, perigee_alt, radius)
    a = compute_semi_major_axis(apogee_alt, perigee_alt, radius)
    theta = np.linspace(0, 2 * np.pi, n_samples)
    x = a * (e + np.cos(theta))/(1 + e * np.cos(theta))
    y = (a * (1 - e ** 2 )/(1 + e * np.cos(theta)))* np.sin(theta)

    #plots graph with features
    fig, ax = plt.subplots()
    ax.plot(x, y, label='Orbit Path')
    ax.set_aspect('equal')
    ax.set_xlabel('x (km)')
    ax.set_ylabel('y (km)')
    ax.legend()
    plt.show()

#TASK 4 - Find largest by mass 
def find_largest_by_mass(satdb):
    """Return largest satellite based on launch mass.
    Parameters
    ----------
    satdb : SatDB
        A satellite database object
    Return
    ------
    largest_mass_sat_info_tuple: tuple
        A tuple of form (idx -> int, name -> str, launch_mass -> float) in that order
    """
    num_of_sat = satdb.get_number_satellites()
    max_sat_mass = 0
    
    #loops over each satelites in database, until it has found a sat with none heavier than it.
    for sat_index in range(num_of_sat):
        current_sat_mass = satdb.get_sat_data(sat_index, ['launch_mass'])[0]
        if current_sat_mass > max_sat_mass:
            max_sat_mass = current_sat_mass
            max_sat_index = sat_index

    #returns a tuple with the relevant info of the sat
    largest_mass_sat_info = satdb.get_sat_data(max_sat_index, ['name', 'launch_mass'])
    largest_mass_sat_info_tuple = (max_sat_index, largest_mass_sat_info[0], largest_mass_sat_info[1])
    return largest_mass_sat_info_tuple


#TASK 5 - Count satellites by country - Done
def count_satellites_by_country(satdb):
    """Return satellite count for each country.

    Note that "country" relates to the entries under
    country field in the UCS database. Some entries are
    not countries, strictly speaking, but rather multi-national
    organisations.
    Parameters
    ----------
    satdb : SatDB
        A satellite database object
    Return
    ------
    sat_per_country_dict: dict -> [str: int]
        Keys are countries, values are count of satellites by country
    """
    
    sat_per_country_dict = {}
    num_of_sat = satdb.get_number_satellites()
    
    #runs through all sats and counts the sats
    for sat_index in range(num_of_sat):
        sat_country = satdb.get_sat_data(sat_index, ['country'])[0]
        if sat_country in sat_per_country_dict:
            sat_per_country_dict[sat_country] += 1
        else:
            sat_per_country_dict[sat_country] = 1
    return sat_per_country_dict
        

    
def sort_countries_by_count(sats_by_country):
    """Return a list of countries sorted largest to smallest based on satellite
    → count.
    
    Parameters
    ----------
    sats_by_country : dict
        Dictionary with countries as keys and satellite count as values

    Return
    ------
    sorted_sat_country_list: list
        A sorted list with tuples as items; each tuple is (country, count)
    """
    sat_country_list = list(sats_by_country.items())
    
    #Using the bubble logic code to sort countries by the quantity of sats
    n = len(sat_country_list)
    for i in range(n-1):
        for j in range(0, n-i-1):
            if sat_country_list[j][1] < sat_country_list[j+1][1]:
                sat_country_list[j], sat_country_list[j+1] = sat_country_list[j+1], sat_country_list[j]
    sorted_sat_country_list = sat_country_list
    return sorted_sat_country_list
     

def plot_top_countries_by_count(srtd_count, n):
    """Produce bar chart of top n countries based on satellite count.

    Parameters
    ----------
    srtd_count : list
        Sorted list (largest to smallest) of tuples of form (country, count)
    n : int
        Number of top countries to include in the list
    """
    #sets up graph variables
    top_n_countries = [count_sats_per_country[0] for count_sats_per_country in srtd_count[:n]]
    sats_per_country = [count_sats_per_country[1] for count_sats_per_country in srtd_count[:n]]

    #Graphs variables with features
    plt.bar(top_n_countries, sats_per_country)
    plt.xlabel('Country')
    plt.ylabel('number of satellites')
    plt.xticks(rotation=90)
    plt.title(f'Top {n} "Countries" based on number of satellites operated')
    plt.show()
    
#TASK 6 - Add eccentricity and period to the database - DONE
    
def field_as_ndarray(satdb, field):
    """Return array of values corresponding to field.

    This function only makes sense when the field of
    interest contains numeric values.
    
    Parameters
    ----------
    satdb : SatDB
        A satellite database object
    field : string
        A field of interest (with numeric values)
        
    Return
    ------
    sat_field_array: ndarray(dtype=float)
        An array with numeric values for all satellites
    """

    num_of_sat = satdb.get_number_satellites()
    sat_field_list = []

    #creates an array using all values in a field (column) chronologically
    for sat_index in range(num_of_sat):
        sat_field_entry = satdb.get_sat_data(sat_index, [field])[0]
        sat_field_list.append(sat_field_entry)
    sat_field_array = np.array(sat_field_list)
    return sat_field_array

def add_eccentricity(satdb, radius_earth):
    """Add eccentricity values to satellite database.

    Parameters
    ----------
    satdb : SatDB
        A satellite database object
    radius_earth: float
        Radius of the Earth in km
        
    Return
    ------
    satdb : SatDB
        Modified version with field 'eccentricity' added
    """
    
    apogee_value_array = field_as_ndarray(satdb, "apogee")
    perigee_value_array = field_as_ndarray(satdb, "perigee")

    #calculates the eccentricity and adds it to the database as a field
    calc_eccentricity_array = compute_eccentricity(apogee_value_array, perigee_value_array, radius_earth)
    satdb.add_field_of_data("eccentricity", calc_eccentricity_array)
    return satdb

def add_period(satdb, radius_earth, grav_param):
    """Add period values to satellite database.

    Parameters
    ----------
    satdb : SatDB
        A satellite database object
    radius_earth: float
        Radius of the Earth in km
    grav_param : float
        Gravitational parameter in km^3/s^2

    Return
    ------
    satdb : SatDB
       Modified version with field 'period' added
    """
    
    apogee_value_array = field_as_ndarray(satdb, "apogee")
    perigee_value_array = field_as_ndarray(satdb, "perigee")

    #calculates the period and adds it to the database as a field
    calc_period_array = compute_period(apogee_value_array, perigee_value_array, radius_earth, grav_param)
    satdb.add_field_of_data("period", calc_period_array)
    return satdb

#TASK 7 - Query Database entry
class Query(object):
    """
    This class acts as a query class for a satellite database.
    It is initialized with a SatDB object and a selection tuple.
    """
    
    def __init__(self, satdb, selection):
        """Initialises the database using the main and selection databases

        Parameters
        ----------
        satdb : SatDB
            A satellite database object
        selection : tuple
            A tuple with integer indices to indicate selection of interest for queries

        Return
        ------
        None
        """
        
        self._satdb = satdb 
        self._selection = selection 
        
        
    def get_selection(self):
        """Returns indices of satellites in selection

        Return
        ------
        self._selection: tuple
            indices of satellites in selection
        """
        return self._selection 
    
    def set_selection(self, selection):
        """Set the selection of satellites for queries to operate on.

        Parameters
        ----------
        selection : tuple
            indices of satellites in selection

        Return
        ------
        None
        """
        self._selection = selection 
                

    def find_range(self, field):
        """Return index and value of minimum and maximum in a field with numeric
        ,→ values.
        
        Parameters
        ----------
        field : string
            Field of interest for determining range
        
        Return
        ------
        range_of_selection_data: tuple, tuple
            Tuples of form (min_index, min_value), (max_index, max_value)
        """

        selection_of_sats_list = []
        
        #adds the valid entries to a list from all sats (in selection)
        for sat_index in self._selection: 
            selection_of_sats_entry = self._satdb.get_sat_data(sat_index, [field])[0] 
            if selection_of_sats_entry != -1: 
                selection_of_sats_list.append(selection_of_sats_entry) 

        #find the range of the data by retrieving min/max indices
        minimum_index = np.argmin(selection_of_sats_list)
        maximum_index = np.argmax(selection_of_sats_list)
        range_of_selection_data = (minimum_index, selection_of_sats_list[minimum_index]), (maximum_index, selection_of_sats_list[maximum_index])
        return range_of_selection_data

    def count_unique_entries(self, field):
        """Count the unique entries in a field with string values.

        Parameter
        ---------
        field : string
            Field of interest for counting unique entries

        Return
        ------
        count_unique_entries: int
            Count of unique entries
        """
        selection_of_sats_list = []
        
        #retrieves the value of a certain field for each sat 
        for sat_index in self._selection:
            selection_of_sats_entry = self._satdb.get_sat_data(sat_index, [field])[0]
            if selection_of_sats_list != -1:
                selection_of_sats_list.append(selection_of_sats_entry)
                
        #counts num. of unique entries
        unique_entry_list = set(selection_of_sats_list)
        count_unique_entries = len(unique_entry_list)
        return count_unique_entries
    
    def compute_statistics(self, field):
        """Return mean and standard deviation for a field of numeric values.

        Parameters
        ----------
        field : string
            Field of interest for statistic calculations.
        
        Return
        ------
        sats_combined_stats: tuple(float, float)
            Tuple contains (mean, std. deviation)
        """
        selection_of_sats_list = []
        
        #retrieves the value of a certain field for each sat 
        for sat_index in self._selection:
            selection_of_sats_entry = self._satdb.get_sat_data(sat_index, [field])[0]
            if selection_of_sats_entry != -1:
                selection_of_sats_list.append(selection_of_sats_entry)

        #returns statistics
        mean = np.mean(selection_of_sats_list)
        standard_deviation = np.std(selection_of_sats_list)
        sats_combined_stats = (mean, standard_deviation)
        return sats_combined_stats

    #TASK 8 - Two more methods for Query - find exact match and filter by range
    def find_exact_match(self, field, str_to_match):
        """Return Query containing satellites with exact string match in given field.
        
        Parameters
        ----------
        field : string
            Field of interest when searching for match.
        str_to_match : string
            String used in match.
        
        Return
        ------
        Query(self._satdb, matching_selections_tuple): Query
            A query object with selection set to only those satellites that match.
        """
        selection_of_sats_list = []

        #finds the matching sats from db
        for sat_index in self._selection:
            sat_field_string = self._satdb.get_sat_data(sat_index, [field])[0]
            if sat_field_string == str_to_match:
                selection_of_sats_list.append(sat_index)
                
        #return matches
        matching_selections_tuple = tuple(selection_of_sats_list)
        return Query(self._satdb, matching_selections_tuple)
                
    def filter_by_range(self, field, min_val, max_val):
        """Return Query containing satellites that fall in given range (of numeric
        ,→ values)
        Valid values in range are: min_val <= val < max_val
        
        Parameters
        ----------
        field : string
            Field of interest when searching for match.
        min_val : float
            Lower bound for range search
        max_val : float
            Upper bound for range search
        
        Return
        ------
        Query(self._satdb, selection_of_sats_list): Query
            A query object with selection set to those satellites that fall in range.
        """
        selection_of_sats_list = []

        #adds all sats with select fields between the range to a list
        for sat_index in self._selection:
            sat_field_entry = self._satdb.get_sat_data(sat_index, [field])[0]
            if min_val <= sat_field_entry < max_val:
                selection_of_sats_list.append(sat_index)
        return Query(self._satdb, selection_of_sats_list)
        
        



#BONUS TASK 9 - Plot the distribution of masses in each of the orbits
def plot_mass_distributions(satdb, radius_earth, grav_param, e_limit, orbits):
    """Produce boxplot of mass distributions, grouped by orbit type.

    Parameters
    ----------
    satdb : SatDB
    A satellite database object
    radius_earth: float
    Radius of the Earth in km
    grav_param : float
    Gravitational parameter in km^3/s^2
    e_limit : float
    Criterion for selecting satellites: e_sat < e_limit
    orbits : dict
    Dictionary with entries: {orbit_abbr: (T_min, T_max)}
    """
    
    leo_sat_mass_list = []
    meo_sat_mass_list = []
    geo_sat_mass_list = []
    num_of_sats = satdb.get_number_satellites()
    
     
    for sat_index in range(num_of_sats):
        #for each sat, calculates/retrieves all the orbital parameters necessary
        alt_of_apo = (satdb.get_sat_data(sat_index, ['apogee'])[0])
        alt_of_peri = (satdb.get_sat_data(sat_index, ['apogee'])[0])
        orbit_period = compute_period(alt_of_apo, alt_of_peri, radius_earth, grav_param)
        orbit_eccent = compute_eccentricity(alt_of_apo, alt_of_peri, radius_earth)
        full_sat_mass = (satdb,get_sat_data(sat_index, ['launch_mass'])[0])

        #Places sat in specific list categorised by orbit 
        if orbit_eccent < e_limit:
            if orbit_period < orbits['LEO'][1]:
                leo_sat_mass_list.append(full_sat_mass)
            elif orbits['MEO'][0] < orbit_period < orbits['MEO'][1]:
                meo_sat_mass_list.append(full_sat_mass)
            elif orbits['GEO'][0] < orbit_period < orbits['GEO'][1]:
                geo_sat_mass_list.append(full_sat_mass)
                
    #graphs box plot with specific features
    sat_mass_arrays = [np.array(leo_sat_mass_list), np.array(meo_sat_mass_list), np.array(geo_sat_mass_list)]
    fig, ax = plt.subplots()
    ax.boxplot(sat_mass_arrays, labels=orbits.keys(), showfliers=False)
    ax.set_title('Mass Distributions of Sats by Type of Orbit')
    ax.set_xlabel('Orbital Type')
    ax.set_ylabel('Sat Mass (kg)')
    plt.show()
