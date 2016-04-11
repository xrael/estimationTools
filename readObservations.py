######################################################
# Functions to read observation files.
#
# Manuel F. Diaz Ramos
######################################################

import csv
import numpy as np

def readRangeRangeRateCSV(obs_file, startTime, DCO = None):
    """
    Read Range and Range-rate from 3 stations from a CSV. Assumes the following columns format:
    [time, range gs1, range gs2, range gs3, range rate gs1, range rate gs2, range rate gs3]
    :param obs_file:
    :param startTime: Start reading data at this time.
    :param DCO: Data cut-off time. Indicates whn to stop reading data.
    :return:
    """
    obs_time_vec = []               # Time vector
    obs_vec = []                    # Vector with observations
    obs_station_nmbr_vec = []       # Number of station for each observation

    with open(obs_file, 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        first_row = True
        for row in csv_reader:
            if first_row:
                first_row = False
                continue

            obs_time = float(row[0])

            if obs_time < startTime:
                continue

            if DCO is not None:
                if obs_time >= DCO: # Only process data up to the DCO
                    break

            if row[1].replace(" ", "") != 'NaN' and row[4].replace(" ", "") != 'NaN': # DSS34 data
                obs_range = float(row[1])
                obs_range_rate = float(row[4])
                obs_time_vec.append(obs_time)
                obs_vec.append([obs_range, obs_range_rate])
                obs_station_nmbr_vec.append(0)

            if row[2].replace(" ", "") != 'NaN' and row[5].replace(" ", "") != 'NaN': # DSS65 data
                obs_range = float(row[2])
                obs_range_rate = float(row[5])
                obs_time_vec.append(obs_time)
                obs_vec.append([obs_range, obs_range_rate])
                obs_station_nmbr_vec.append(1)

            if row[3].replace(" ", "") != 'NaN' and row[6].replace(" ", "") != 'NaN': # DSS13 data
                obs_range = float(row[3])
                obs_range_rate = float(row[6])
                obs_time_vec.append(obs_time)
                obs_vec.append([obs_range, obs_range_rate])
                obs_station_nmbr_vec.append(2)

    obs_time_vec = np.array(obs_time_vec)
    obs_vec = np.array(obs_vec)
    obs_station_nmbr_vec = np.array(obs_station_nmbr_vec)

    return (obs_time_vec, obs_vec, obs_station_nmbr_vec)