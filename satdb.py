# Author: Rowan J. Gollan
"""Module to provide a class for interacting with satellite databases.

Classes:
    SatDB
"""

import csv
import warnings
from datetime import date

conv_fn = {
    'name': str,
    'country': str,
    'operator': str,
    'users': str,
    'purpose': str,
    'perigee': float,
    'apogee': float,
    'eccentricity': float,
    'launch_mass': float,
    'date_of_launch': date.fromisoformat,
    'expected_lifetime': float
    }


def warn_caller(message: str, category=None, stacklevel=3):
    """Wrapper around UserWarning that has stacklevel of 3"""
    warnings.warn(message, category, stacklevel)


def simple_format(message, category, filename, lineno, line=None):
    if line is None:
        import linecache
        line = linecache.getline(filename, lineno).strip()
        
    return (
        "SatDB Warning:\n"
        f"  File: \"{filename}\", line {lineno}\n    {line}" 
        f"\n{category.__name__}: {message}\n"
    )


warnings.formatwarning = simple_format
# NOTE: For testing, use warnings.simplefilter("ignore")


class SatDB(object):
    """Class to provide interaction with a satellite database.

    Methods
    -------
    get_number_of_satellites
    get_sat_data
    add_field_of_data
    """
    
    def __init__(self, filename):
        """Construct a SatDB object.

        Parameters
        ----------
        filename : string
           A CSV file with data for satellites.
        """
        with open(filename, 'r') as sat_csv_file:
            data_list = list(csv.reader(sat_csv_file, delimiter=','))
        # parse header to get fields in data file
        self._fields = data_list[0].copy()
        self._fields.pop() # remove trailing empty field
        # assemble data as list of dictionaries, each dictionary represents satellite data
        self._data = []
        for row in data_list[1:]:
            self._data.append({})
            for pos, field in enumerate(self._fields):
                self._data[-1][field] = conv_fn[field](row[pos])

    def get_number_satellites(self):
        """Returns the number of satellites in the database."""
        return len(self._data)

    def get_sat_data(self, idx, fields):
        """Returns a selection of data for a given satellite.

        The caller provides a satellite id and list of fields.
        The method returns data associated with those fields for the given satellite.
        If the satellite is not present in database, or a specific field is not available,
        then an empty tuple is returned.
        If even one field is missing, the entire query is returned as empty.
        
        Parameters
        ----------
        idx : int
           Index of satellite in database.
        fields : list[string]
           List of strings; each string is a desired field from database for given satellite

        Returns
        -------
        tuple
           Tuple contains values in order of provided list of fields
           An empty tuple indicates that no data could be found

        
        """
        # Hand back an empty tuple for all early exits
        if not (0 <= idx < self.get_number_satellites()):
            warn_caller(f"Satellite index {idx} out of range")
            return tuple()

        if not isinstance(fields, (tuple, list)):
            warn_caller("Must pass a list or tuple of fields")
            return tuple()

        values = []
        for field in fields:
            if field in self._data[idx]:
                values.append(self._data[idx][field])
            else:
                # We've hit an invalid field, so entire query is invalid
                warn_caller(f"Database does not contain field '{field}'")
                return tuple()
        return tuple(values)

    def add_field_of_data(self, field, data):
        """Adds a new field to satellite database

        The caller provides a field name and an iterable of data (eg. list or numpy array).
        This is added to the database.
        If field name is already present in database, then this method OVERWRITES existing
        data with what is supplied here.
        The amount of data must match number of satellites in database.
        If it does not match, then no action is performed.
        A warning is given and the method returns without altering the database.
        
        Parameters
        ----------
        field : string
           Name of field for additional data
        data : list|tuple|ndarray
           An iterable object containing data for each satellite
        """
        n_sats = self.get_number_satellites()
        if len(data) != n_sats:
            warn_caller(
                "Could not add field data because amount of data is incorrect "
                f"(expected {n_sats}, got {len(data)})"
            )
            return
        if not isinstance(field, str):
            warn_caller("Only a single field of data can be added at a time")
            return 
        for i in range(n_sats):
            self._data[i][field] = data[i]
        return


# Restrict star (*) import to the SatDB class
__all__ = ["SatDB"]
        
if __name__ == '__main__':
    sat_db = SatDB('test.csv')
    n_sats = sat_db.get_number_satellites()
    print(f"number of satellites in database: {n_sats}")
    sat_id = 23
    (operator, launch_mass) = sat_db.get_sat_data(sat_id, ['operator', 'launch-mass'])
    print(f"id: {sat_id}  operator: {operator}  launch-mass: {launch_mass}")
    
