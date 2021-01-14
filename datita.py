import string
import numpy
import math
from datetime import datetime, timedelta, date, time
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec

global birthdate, last_date, all_dates, n_days, n_days_sleep
global morning, night, label_type

def parse_date(date_string):
    # convert Hatch's time and date strings to datetime objects

    date = datetime.strptime(date_string, '%m/%d/%Y %I:%M %p')
    return date


def parse_time(time_datetime):
    # convert datetime time objects to fractional time / 24 hours
    
    time_fractional = time_datetime.hour + time_datetime.minute/60.
    return time_fractional


def read_data(filename):
    # read a specified Hatch export file and assign lines to appropriate categories
    
    f = open(filename, 'r')
    lines = f.readlines()

    line_types = ['lines_feeding', 'lines_diapers', 'lines_lengths', 'lines_sleeps', 'lines_weights', 'lines_notes']
    type_start = [0, 0, 0, 0, 0, 0]
    i_type = 0

    for i in range(len(lines)):
        if '=== BEGIN' in lines[i]:
            type_start[i_type] = int(i+1)
            i_type = i_type + 1    

    lines_feedings  = lines[type_start[0]:type_start[1]-2]
    lines_diapers   = lines[type_start[1]:type_start[2]-2]
    lines_lengths   = lines[type_start[2]:type_start[3]-2]
    lines_sleeps    = lines[type_start[3]:type_start[4]-2]
    lines_weights   = lines[type_start[4]:type_start[5]-2]
    lines_notes     = lines[type_start[5]:-1]

    return lines_sleeps, lines_feedings, lines_diapers, lines_weights, lines_lengths


def define_months():
    n_months = math.floor((last_date - birthdate).days/30)+2
    month_dates = []
    for i in range(n_months):
        month_dates.append(datetime.combine(birthdate + timedelta(days=30*i), time(0,0)))
    return n_months, month_dates


def read_percentiles(percentile_type, gender, n_months):
    # read CDC percentile tables 

    f = open('percentiles_{type}_{gender}.txt'.format(type=percentile_type, gender=gender), 'r')
    lines = f.readlines()
    
    if percentile_type == 'weight-length':
        all_lengths = numpy.zeros(len(lines))
        max_length = math.ceil(n_months) + 1

        for i in range(len(lines)):
            percentiles_data = lines[i].split()
            all_lengths[i] = float(percentiles_data[0])*0.393701
            if all_lengths[i] < max_length: 
                j = i

        lengths = numpy.zeros(j)
        weight_percentiles = numpy.zeros((9, j))

        for i in range(j):
            percentiles_data = lines[i].split()
            lengths[i] = all_lengths[i]
            
            for j in range(9):
                weight_percentiles[j][i] = float(percentiles_data[j+1])*2.20462

        return lengths, weight_percentiles

    else:

        percentiles = numpy.zeros((9, n_months))

        for i in range(n_months):
            month_percentiles = lines[i].split()
            for j in range(9):
                if percentile_type == 'weight':
                    percentiles[j][i] = float(month_percentiles[j])*2.20462
                elif percentile_type == 'length':
                    percentiles[j][i] = float(month_percentiles[j])*0.393701
                elif percentile_type == 'head':
                    percentiles[j][i] = float(month_percentiles[j])*0.393701

        return percentiles


def parse_sleep(lines_sleeps):
    # parse Hatch sleep data lines

    data_sleep = [] # start, duration

    # record start time and duration of sleep from midnight to midnight    
    data_sleep_24 = [] # start, end, duration

    for i in range(len(lines_sleeps)):
        data = lines_sleeps[i].split(',')
        
        start = parse_date(data[1])
        end = parse_date(data[2])

        if end.day != start.day: 
            # fix sleeps to end at midnight
            end_1 = datetime(start.year, start.month, start.day, 23, 59)
            start_2 = datetime(end.year, end.month, end.day, 0, 0)
            duration_1 = (end_1 - start).seconds/60.
            duration_2 = (end - start_2).seconds/60.
            data_sleep_24.append([start, end_1, duration_1])
            data_sleep_24.append([start_2, end, duration_2])
            data_sleep.append([start, duration_1 + duration_2])
        else: 
            duration = (end - start).seconds/60.
            data_sleep_24.append([start, end, duration])
            data_sleep.append([start, duration])

    return data_sleep, data_sleep_24
        

def parse_feedings(lines_feeding):

    data_feeding = []

    for i in range(len(lines_feeding)):
        data = lines_feeding[i].split(',')
        start = parse_date(data[1])
        end = parse_date(data[2])
        amount = float(data[4][:-2])

        data_feeding.append([start, end, amount])

    return data_feeding
        

def parse_weights(lines_weights, month_dates):
    # I distinguish between "official weights" (from doctor visits)
    # and weights we take at home (on the Hatch, always in clothes and a clean diaper)

    wdates_all = []
    weights_all = []

    data_weight_official = []

    for i in range(len(lines_weights)):
        data = lines_weights[i].split(',')
        date = parse_date(data[1]).date()
        
        weight = data[4]
        lboz = weight.split(' ')
        lb = float(lboz[0][:-2])
        if len(lboz) == 1:
            oz = 0.
        else:
            oz = float(lboz[1][:-2])

        if len(data[8]) > 1:
            date = datetime.combine(date, time(0, 0))
            data_weight_official.append([date, lb + oz/16.])
        else: 
            wdates_all.append(parse_date(data[1]).date())
            weights_all.append(lb + oz/16.)

    # calculate one weight per day
    data_weight = [[] for i in range(len(all_dates)+1)] 
    for i in range(len(weights_all)):
        for j in range(len(all_dates)):
            if all_dates[j] == wdates_all[i]:
                date = datetime.combine(wdates_all[i], time(0, 0))
                data_weight[j] = [date, weights_all[i]]
    data_weight[-1] = [month_dates[-1], -1]
    
    # fill in weights on days without
    for i in range(len(data_weight)):
        if data_weight[i] == []:
            date = datetime.combine(all_dates[i], time(0, 0))
            data_weight[i] = [date, data_weight[i-1][1]]

    return data_weight, data_weight_official


def parse_lengths(lines_lengths, month_dates):
    data_length = []

    for i in range(len(lines_lengths)):
        data = lines_lengths[i].split(',')
        date = parse_date(data[1])
        length = float(data[4][0:-2])
        data_length.append([date, length])
    data_length.append([month_dates[-1], -1])
        
    return data_length


def parse_head(head_file):
    f = open(head_file, 'r')
    lines_head = f.readlines()

    data_head = []

    for i in range(len(lines_head)):
        data = lines_head[i].split()
        head_date = datetime(int(data[0]), int(data[1]), int(data[2]), 0, 0)
        head = float(data[3])
        data_head.append([head_date, head])
    data_head.append([month_dates[-1], -1])
        
    return data_head


def parse_dirty_diapers(lines_diapers):

    dirty_date = []

    for i in range(len(lines_diapers)):
        data = lines_diapers[i].split(',')
        diaper_date = parse_date(data[1])
        diaper_type = data[7]

        if diaper_type in ['Both', 'Dirty']:
            diaper_type = True
            dirty_date.append(diaper_date)
    
    # separate diapers by day
    data_diapers = []
    for i in range(n_days+1):
        new_date = datetime.combine(birthdate + timedelta(days=i), time(0,0))
        data_diapers.append([new_date, []])

    j = 0 # index of original data
    k = 0 # index of date
    while j < len(dirty_date):
        if dirty_date[j].date() == data_diapers[k][0].date():
            data_diapers[k][1].append(parse_time(dirty_date[j].time()))
            j = j + 1
        else: 
            k = k + 1
    
    return data_diapers


def write_date_tick_labels(date_tick_labels, the_datetime, week=None, month=None):

    if label_type == 'week-date':
        date_tick_labels.append('{month:d}/{day:d}'.format(month=the_datetime.month, day=the_datetime.day))
    elif label_type == 'week':
        date_tick_labels.append(week)
    elif label_type == 'month':
        date_tick_labels.append(month)

    return date_tick_labels


def date_ticks(data, label_type):

    # determine starting week
    starting_week = 0
    for i in range(100):
        if birthdate + timedelta(weeks=i) <= data[0][0].date():
            starting_week = i

    # determine starting week
    starting_month = 0
    for i in range(12):
        if birthdate + timedelta(days=i*30) <= data[0][0].date():
            starting_month = i

    if label_type in ['week', 'week-date']:
        date_ticks = [birthdate + timedelta(weeks=starting_week)]
        unit_increment = timedelta(weeks=1)
    elif label_type == 'month':
        date_ticks = [birthdate + timedelta(days=starting_month*30)]
        unit_increment = timedelta(days=30)

    date_tick_labels = []
    date_tick_labels = write_date_tick_labels(date_tick_labels, date_ticks[0], week=starting_week, month=starting_month)

    for i in range(100):
        if date_ticks[-1] + unit_increment <= data[-1][0].date():
            new_date = date_ticks[-1] + unit_increment
            date_ticks.append(new_date)

            date_tick_labels = write_date_tick_labels(date_tick_labels, new_date, week = starting_week + i + 1, month = starting_month + i + 1)

    new_date = date_ticks[-1] + unit_increment
    date_ticks.append(new_date)
    date_tick_labels = write_date_tick_labels(date_tick_labels, new_date, week = '', month = '')

    if label_type == 'week':
        plt.xlabel('weeks of age')
    if label_type == 'month':
        plt.xlabel('months of age')

    plt.gca().tick_params(axis = 'x', which = 'major', labelsize = 8)
    plt.xticks(date_ticks, rotation=45)
    plt.gca().set_xticklabels(date_tick_labels)
    plt.gca().set_xlim([date_ticks[0], date_ticks[-1]])

    return date_ticks, date_tick_labels


def time_ticks(axis):
    time_ticks = [0, 4, 8, 12, 16, 20, 24]
    time_tick_labels = ['midnight', '4am', '8am', 'noon', '4pm', '8pm', 'midnight']

    if axis == 'x':
        plt.gca().set_xlim([0, 24])
        plt.xticks(time_ticks)
        plt.gca().set_xticklabels(time_tick_labels)
        plt.axvspan(xmin=0, xmax=morning, color=colormap(1.0), alpha=0.1)
        plt.axvspan(xmin=morning, xmax=night, color=colormap(0.0), alpha=0.1)
        plt.axvspan(xmin=night, xmax=24, color=colormap(1.0), alpha=0.1)
    elif axis == 'y':
        plt.gca().set_ylim([0, 24])
        plt.yticks(time_ticks)
        plt.gca().set_yticklabels(time_tick_labels)
        plt.axhspan(ymin=0, ymax=morning, color=colormap(1.0), alpha=0.1)
        plt.axhspan(ymin=morning, ymax=night, color=colormap(0.0), alpha=0.1)
        plt.axhspan(ymin=night, ymax=24, color=colormap(1.0), alpha=0.1)


def plot_sleep(data_sleep, n_days_sleep):

    norm = matplotlib.colors.Normalize(vmin=0, vmax=n_days_sleep)

    # sleep duration histograms
    duration = []
    for i in range(len(data_sleep)):
        duration.append(data_sleep[i][1])
    weights = numpy.ones_like(duration)/len(duration)*100.
    
    plt.figure()
    plt.suptitle('Sleep time and duration')
    gridspec.GridSpec(3,3)
    
    plt.subplot2grid((3,3), (0,0), colspan=2, rowspan=3)
    time_ticks('x')
    plt.ylabel('hours')
    plt.yticks([0., 60., 120., 180., 240., 300., 360., 420., 480.])
    plt.gca().set_yticklabels([0, 1, 2, 3, 4, 5, 6, 7, 8])
    plt.ylim([0, 488])
    for i in range(len(data_sleep)):
        start_time = parse_time(data_sleep[i][0])
        color = colormap(norm((data_sleep[i][0] - data_sleep[0][0]).days))
        if (data_sleep[-1][0] - data_sleep[i][0]).days < 8:
            alpha = 1
        else: 
            alpha = 0.25
        plt.plot(start_time, data_sleep[i][1], color=color, marker='o', alpha=alpha)

    plt.subplot2grid((3,3), (0,2), colspan=1, rowspan=3)
    plt.ylim([0, 488])
    plt.yticks([0., 60., 120., 180., 240., 300., 360., 420., 480.])
    plt.gca().set_yticklabels([])
    plt.hist(duration, bins=numpy.arange(-10., 490., 20), weights=weights, histtype='stepfilled', color='k', orientation='horizontal')

    plt.savefig('sleep_duration')

    # max interrupted sleep
    plt.figure()
    plt.suptitle('Maximum uninterrupted sleep')
    gridspec.GridSpec(3,3)

    max_sleep = [[data_sleep[0][0], 0.]]
    for i in range(len(data_sleep)):
        if data_sleep[i][0].date() == max_sleep[-1][0].date():
            if data_sleep[i][1] > max_sleep[-1][1]:
                max_sleep[-1] = [data_sleep[i][0], data_sleep[i][1]]
        else: 
            max_sleep.append(data_sleep[i])

    plt.subplot2grid((4,4), (1,0), colspan=3, rowspan=3)
    time_ticks('x')
    plt.ylabel('hours')
    plt.yticks([0., 60., 120., 180., 240., 300., 360., 420., 480.])
    plt.gca().set_yticklabels([0, 1, 2, 3, 4, 5, 6, 7, 8])
    plt.ylim([0, 488])
    durations = []
    times = []
    for i in range(len(max_sleep)):
        date = max_sleep[i][0]
        time = parse_time(max_sleep[i][0].time())
        duration = max_sleep[i][1]
        times.append(time)
        durations.append(duration)
        color = colormap(norm((date - max_sleep[0][0]).days))
        plt.plot(time, duration, 'o', color=color)

    plt.subplot2grid((4,4), (1,3), colspan=1, rowspan=3)
    plt.ylim([0, 480.])
    plt.yticks([0., 60., 120., 180., 240., 300., 360., 420., 480.])
    plt.gca().set_yticklabels([])
    weights = numpy.ones_like(durations)/len(durations)*100.
    plt.hist(durations, bins=numpy.arange(-15., 495., 30), weights=weights, histtype='stepfilled', color='k', orientation='horizontal')

    plt.subplot2grid((4,4), (0,0), colspan=3, rowspan=1)
    weights = numpy.ones_like(times)/len(times)*100.
    plt.hist(times, bins=numpy.arange(-0.5, 24.5, 1.), weights=weights, histtype='stepfilled', color='k')
    time_ticks('x')
    plt.gca().set_xticklabels([])

    plt.savefig('sleep_max.png')

    # wake windows
    plt.figure()
    plt.suptitle('Wake windows')
    gridspec.GridSpec(3,3)

    data_wake = []
    for i in range(len(data_sleep)-1):
        start = data_sleep[i][0] + timedelta(minutes=math.floor(data_sleep[i][1]))
        end = data_sleep[i+1][0]
        wake_window = (end - start).seconds/60.
        data_wake.append([start, wake_window])
    
    plt.subplot2grid((3,3), (0,0), colspan=2, rowspan=3)
    time_ticks('x')
    plt.ylabel('hours')
    plt.yticks([0., 60., 120., 180., 240., 300.])
    plt.gca().set_yticklabels([0, 1, 2, 3, 4, 5])
    plt.ylim([0, 300])
    wakes = []
    for i in range(len(data_wake)):
        time = parse_time(data_wake[i][0].time())
        color = colormap(norm((data_wake[i][0] - max_sleep[0][0]).days))
        if (data_wake[-1][0] - data_wake[i][0]).days < 8:
            alpha = 1
        else: 
            alpha = 0.25
        plt.plot(time, data_wake[i][1], 'o', color=color, alpha=alpha)
        wakes.append(data_wake[i][1])

    plt.subplot2grid((3,3), (0,2), colspan=1, rowspan=3)
    plt.ylim([0, 300.])
    plt.yticks([0., 60., 120., 180., 240., 300.])
    plt.gca().set_yticklabels([])
    weights = numpy.ones_like(wakes)/len(wakes)*100.
    plt.hist(wakes, bins=numpy.arange(-15., 315., 30), weights=weights, histtype='stepfilled', color='k', orientation='horizontal')

    plt.savefig('sleep_wake.png')

    # bedtimes and waketimes
    plt.figure()
    [bedtimes, waketimes] = bed_wake_times(data_sleep, data_wake)

    plt.subplot2grid((3,3), (0,0), colspan=2, rowspan=3)
    times_bed = []
    times_wake = []
    for i in range(len(bedtimes)):
        times_bed.append(bedtimes[i][1])
        times_wake.append(waketimes[i][1])
        plt.plot(bedtimes[i][0], bedtimes[i][1], 'o', color=colormap(1.0))
        plt.plot(waketimes[i][0], waketimes[i][1], 'o', color=colormap(0.5))
    date_ticks(data_sleep, label_type)
    time_ticks('y')

    plt.subplot2grid((3,3), (0,2), colspan=1, rowspan=3)
    plt.ylim([0, 24.])
    time_ticks('y')
    plt.gca().set_yticklabels([])
    weights = numpy.ones_like(times_bed)/len(times_bed)*100.
    plt.hist(times_bed, bins=numpy.arange(0., 24., 0.5), weights=weights, histtype='stepfilled', color=colormap(1.0), orientation='horizontal')
    weights = numpy.ones_like(times_wake)/len(times_wake)*100.
    plt.hist(times_wake, bins=numpy.arange(0., 24., 0.5), weights=weights, histtype='stepfilled', color=colormap(0.5), orientation='horizontal', alpha=0.5)

    plt.savefig('sleep_bed_wake_times.png')

    # correlation between bedtime and wakeup time 
    plt.figure()
    for i in range(len(bedtimes)-1):
        recency = (bedtimes[i][0] - bedtimes[0][0]).days
        alpha = 1.0
        if recency < len(bedtimes)-7:
            alpha = 0.5
        color = colormap(norm(recency))
        plt.plot(bedtimes[i][1], waketimes[i+1][1], 'o', color=color, alpha=alpha)
    
    plt.axis([18., 24., 6., 10.])
    plt.xlabel('bedtime')
    plt.ylabel('waketime')
    plt.savefig('sleep_bed_wake_correlation')
    return bedtimes, waketimes


def determine_bedtime(bedtimes, current_date_sleep):

    bedtime_index = -1
    
    # look for first sleep after bedtime; default is midnight
    for j in range(1,len(current_date_sleep)+1):
        if current_date_sleep[-j][0] > night:
            bedtime_index = -j

    # check for long wake period following (or if sleep is less than 1.5 hours?)
    if bedtime_index < -1:
        wake_period_after = current_date_sleep[bedtime_index+1][0] - current_date_sleep[bedtime_index][1]
        if wake_period_after > 0.5:
            bedtime_index = bedtime_index + 1

    # check for short wake period before 
    wake_period_before = current_date_sleep[bedtime_index][0] - current_date_sleep[bedtime_index-1][1] 
    if wake_period_before < 1.:
        bedtime_index = bedtime_index - 1

    # check if bedtime was really midnight or later (set to midnight)
    last_wakeup = current_date_sleep[-1][1]
    if last_wakeup < 23:
        bedtimes[-1][1] = 24.
    else:
        bedtimes[-1][1] = current_date_sleep[bedtime_index][0]

    return bedtimes


def determine_waketime(waketimes, current_date_wake):
    # only problem is 10/20, when there were two short wakeups

    # look for first wakeup after one hour before the day starts
    for j in range(1,len(current_date_wake)+1):
        if current_date_wake[-j][0] > morning-1.5:
            waketime_index = len(current_date_wake)-j

    # check for short wake period following 
    wake_period_after = current_date_wake[waketime_index][1] - current_date_wake[waketime_index][0]
    if wake_period_after < 0.8:
        waketime_index = waketime_index + 1

    # check for long wake period before
    if waketime_index > 0:
        wake_period_before = current_date_wake[waketime_index-1][1] - current_date_wake[waketime_index-1][0] 
        if wake_period_before > 1.5 and current_date_wake[waketime_index-1][0] > 4.:
            waketime_index = waketime_index - 1

    waketimes[-1][1] = current_date_wake[waketime_index][0]

    return waketimes


def bed_wake_times(data_sleep, data_wake):
    # bedtime
    bedtimes = [[data_sleep[0][0].date(), 24.0]]
    current_date_sleep = []

    for i in range(len(data_sleep)):

        sleep_time = parse_time(data_sleep[i][0])
        wake_time = sleep_time + data_sleep[i][1]/60.
        if data_sleep[i][0].date() == bedtimes[-1][0]:
            # gather all of a day's sleep 
            current_date_sleep.append([sleep_time, wake_time])
            
        else: # at the end of the day
            bedtimes = determine_bedtime(bedtimes, current_date_sleep)
            # plt.plot(bedtimes[-1][0], bedtimes[-1][1], 'o', color=colormap(1.0))

            # set up next day
            bedtimes.append([data_sleep[i][0].date(), 24.0])
            current_date_sleep = [[sleep_time, wake_time]]

    # waketime
    waketimes = [[data_wake[0][0].date(), 0.0]]
    current_date_wake = []

    for i in range(len(data_wake)):

        wake_time = parse_time(data_wake[i][0])
        sleep_time = wake_time + data_wake[i][1]/60.
        if data_wake[i][0].date() == waketimes[-1][0]:
            # gather all of a day's wakeups 
            current_date_wake.append([wake_time, sleep_time])
            
        else: # at the end of the day
            waketimes = determine_waketime(waketimes, current_date_wake)
            # plt.plot(waketimes[-1][0], waketimes[-1][1], 'o', color=colormap(0.5))

            # set up next day
            waketimes.append([data_wake[i][0].date(), 24.0])
            current_date_wake = [[wake_time, sleep_time]]

    return bedtimes, waketimes


def plot_sleep_24(data_sleep_24, bedtimes, waketimes):
    # sleep schedule with day and night distinguished
    plt.figure()
    plt.title('Sleep schedule')
    [week_dates, week_names] = date_ticks(data_sleep_24, 'week')
    time_ticks('y')

    day_index = 0
    for i in range(1, len(data_sleep_24)):
        day = data_sleep_24[i][0].date()
        if day != data_sleep_24[i-1][0].date():
            day_index = day_index + 1
        sleep_time = parse_time(data_sleep_24[i][0])
        wake_time = parse_time(data_sleep_24[i][1])

        if sleep_time >= bedtimes[day_index][1] or wake_time <= waketimes[day_index][1] + 0.1:
            color = colormap(1.0)
        else:
            color = colormap(0.4)

        plt.plot([day, day], [sleep_time, wake_time], '-', linewidth=4, color=color)
        day = day + timedelta(days=-1)
        plt.plot([day, day], [sleep_time+24, wake_time+24], '-', linewidth=4, color=color)

    plt.gca().set_ylim([0, 36])
    plt.savefig('sleep_schedule.png')

    # heat map of sleep schedule by week
    plt.figure()
    plt.title('Sleep schedule')
    time_ticks('y')
    plt.xlabel('weeks')

    j = 0
    for i in range(len(data_sleep_24)):
        date = data_sleep_24[i][0].date()
        if date >= week_dates[j+1]:
            j = j + 1
        x = j
        y1 = parse_time(data_sleep_24[i][0])
        y2 = parse_time(data_sleep_24[i][1])
        plt.plot([x, x], [y1, y2], 'k-', linewidth=20, alpha=0.2)

    plt.savefig('sleep_schedule_heat.png')


    # total sleep in 24 hours
    total_sleep_24 = [0.]
    sleep_dates = [data_sleep_24[0][0].date()]

    for i in range(len(data_sleep_24)):
        if data_sleep_24[i][0].date() == sleep_dates[-1]:
            total_sleep_24[-1] = total_sleep_24[-1] + data_sleep_24[i][2]/60.
        else: 
            sleep_dates.append(sleep_dates[-1] + timedelta(days=1))
            total_sleep_24.append(data_sleep_24[i][2]/60.)

    plt.figure()
    plt.suptitle('Daily sleep')
    gridspec.GridSpec(3,3)

    plt.subplot2grid((3,3), (0,0), colspan=2, rowspan=3)
    date_ticks(data_sleep_24, label_type)
    plt.ylim([10., 18.])
    plt.ylabel('hours')
    plt.plot(sleep_dates, total_sleep_24, 'k.')

    plt.subplot2grid((3,3), (0,2), colspan=1, rowspan=3)
    plt.ylim([10., 18.])
    weights = numpy.ones_like(total_sleep_24)/len(total_sleep_24)*100.
    plt.hist(total_sleep_24, bins=numpy.arange(9.50, 21.50, 1), weights=weights, histtype='stepfilled', color='k', orientation='horizontal')
    plt.yticks([])

    plt.savefig('sleep_total.png')


def plot_feeding(data_feeding):

    # separate feedings by day
    feeding_dates = [data_feeding[0][0].date()]
    feeding_totals = []
    feeding_amounts = [[]]
    feeding_starts = [[]]
    feeding_ends = [[]]

    for i in range(len(data_feeding)):
        if data_feeding[i][0].date() != feeding_dates[-1]:
            feeding_dates.append(feeding_dates[-1] + timedelta(days=1))
            feeding_amounts.append([])
            feeding_starts.append([])
            feeding_ends.append([])
        feeding_starts[-1].append(parse_time(data_feeding[i][0]))
        feeding_ends[-1].append(parse_time(data_feeding[i][1]))
        feeding_amounts[-1].append(data_feeding[i][2])


    # group nearby feedings (<1hr) into one
    feeding_time = [data_feeding[0][0]]
    feeding_amount = [data_feeding[0][2]]

    for i in range(1, len(data_feeding)):
        if data_feeding[i][0] - data_feeding[i-1][0] < timedelta(hours=1):
            feeding_amount[-1] = feeding_amount[-1] + data_feeding[i][2]
        else: 
            feeding_time.append(data_feeding[i][0])
            feeding_amount.append(data_feeding[i][2])
        
    # feeding kymographs
    plt.figure()
    plt.title('Intake over time')
    time_ticks('x')
    plt.ylabel('ounces')
    norm = matplotlib.colors.Normalize(vmin=0, vmax=n_days)
    
    for i in range(len(feeding_dates)):  
        feeding_totals.append(sum(feeding_amounts[i]))
        x = [0.]
        y = [0.]
        for j in range(len(feeding_amounts[i])):
            x.append(feeding_starts[i][j])
            x.append(feeding_ends[i][j])
            y.append(y[-1])
            y.append(y[-1] + feeding_amounts[i][j])
        if x[-1] < 12: # fix feedings that continue into next day
            x[-1] = 24.
        x.append(24)
        y.append(y[-1])    

        color = colormap(norm((feeding_dates[i] - data_feeding[0][0].date()).days))
        if i < len(feeding_dates) - 1:
            if i > len(feeding_dates) - 8: 
                alpha = 1.
            else: 
                alpha = 0.3
            plt.plot(x, y, linestyle='-', color=color, alpha=alpha)

    plt.plot([0, 24], [0, 24], '--', color='white')
    plt.savefig('feeding_over_time.png')

    # feeding totals
    plt.figure()
    plt.title('Feeding totals')
    plt.ylim([0, 40.])
    plt.ylabel('ounces')
    date_ticks(data_feeding, label_type)
    plt.plot(feeding_dates, feeding_totals, 'k.')
    plt.savefig('feeding_totals.png')

    plt.figure()
    plt.title('Intake as a percentage of weight')
    date_ticks(data_feeding, label_type)
    plt.plot([feeding_dates[0], feeding_dates[-1]], [16, 16], 'k--')
    plt.plot([feeding_dates[0], feeding_dates[-1]], [12.5, 12.5], 'k--')
    for i in range(len(feeding_dates)):
        plt.plot(feeding_dates[i], feeding_totals[i]/data_weight[i][1]/16.*100., 'k.')
    plt.savefig('feeding_percentage.png')

    # amount per feeding
    plt.figure()
    plt.title('Amount per feeding')
    date_ticks(data_feeding, label_type)
    time_ticks('y')
    
    size = [2, 3, 5, 6, 7, 8, 10, 12]
    color = [0.0, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9, 1.0]
    label = ['1oz', '2oz', '3oz', '4oz', '5oz', '6oz', '7oz', '8+oz']

    for i in range(len(feeding_time)): 
        x = feeding_time[i].date()
        y = parse_time(feeding_time[i])
        z = int(math.ceil(feeding_amount[i]))
        if z > len(size): 
            z = len(size)
        if z > 0:
            plt.plot(x, y, marker='o', color=colormap(color[z-1]), markersize=size[z-1])
    
    handles = size_color_legend(size, color, label)
    plt.gca().legend(handles=handles, loc='lower left')    
    plt.savefig('feeding_amount_per.png')


    plt.figure()
    plt.title('Feeding amount vs. interval')
    norm = matplotlib.colors.Normalize(vmin=0, vmax=n_days)
    for i in range(1, len(feeding_time)):
        interval = (feeding_time[i] - feeding_time[i-1]).seconds/3600.
        color = colormap(norm((feeding_time[i].date() - birthdate).days))
        if (feeding_time[-1] - feeding_time[i]).days < 8:
            alpha = 1
        else: 
            alpha = 0.25
        plt.plot(interval, feeding_amount[i], color=color, marker='o', alpha=alpha)

    plt.plot([0, 6], [0, 6], 'k--')
    plt.gca().set_xlim([0, 8])
    plt.xlabel('hours')
    plt.ylabel('ounces')
    plt.savefig('feeding_amount_interval.png')


    # group feedings by time of day
    plt.figure()
    plt.title('Feeding by time of day')
    norm = matplotlib.colors.Normalize(vmin=0, vmax=n_days)
    times = [10., 14., 18., 22., 2., 6.]
    colors = [0.4, 0.0, 0.2, 0.8, 1.0, 0.6]
    labels = ['8am-12', '12-4pm', '4-8pm', '>8pm', '<4am', '4-8am']
    size = [8, 8, 8, 8, 8, 8]
    handles = size_color_legend(size, colors, labels)

    feeding_by_time = numpy.zeros((len(feeding_amounts), 6))
    for i in range(len(feeding_amounts)):
        for j in range(len(feeding_amounts[i])):
            feeding_time = feeding_starts[i][j]
            for k in range(6):
                if feeding_time > times[k] - 2. and feeding_time < times[k] + 2.:
                    feeding_by_time[i][k] = feeding_by_time[i][k] + feeding_amounts[i][j]

    for k in range(6):
        bin_avgs = [0.]
        bin_dates = [feeding_dates[0]]
        for i in range(len(feeding_amounts)):
            bin_avgs[-1] = bin_avgs[-1] + feeding_by_time[i][k]/feeding_totals[i]
            if i%7 == 0:
                bin_avgs[-1] = bin_avgs[-1]/7.
                bin_avgs.append(0.)
                bin_dates.append(feeding_dates[i])
            elif i == len(feeding_amounts)-1:
                bin_avgs[-1] = bin_avgs[-1]/(i%7)
        plt.plot(bin_dates, bin_avgs, 'o--', color=colormap(colors[k]))
    
    plt.gca().legend(handles=handles, loc='lower left')    
    plt.ylim([0, 0.4])
    date_ticks(data_feeding, label_type)
    plt.savefig('feeding_time_of_day.png')


def plot_diapers(data_diapers):
    diaper_times = []
    diaper_dates = []
    n_dirty = []

    for i in range(len(data_diapers)):
        diaper_dates.append(data_diapers[i][0])
        diaper_times = diaper_times + data_diapers[i][1]
        n_dirty.append(len(data_diapers[i][1]))

    plt.figure()
    plt.suptitle('When dirty diapers happen')
    gridspec.GridSpec(3,3)
    
    plt.subplot2grid((3,3), (0,0), colspan=2, rowspan=3)
    time_ticks('y')
    date_ticks(data_diapers, label_type)
    for i in range(len(data_diapers)):
            for j in range(len(data_diapers[i][1])):
                plt.plot(data_diapers[i][0], data_diapers[i][1][j], 'ko')

    plt.subplot2grid((3,3), (0,2), colspan=1, rowspan=3)
    plt.ylim([0, 24])
    plt.yticks([0, 4, 8, 12, 16, 20, 24])
    plt.gca().set_yticklabels([])
    weights = numpy.ones_like(diaper_times)/len(diaper_times)*100.
    plt.hist(diaper_times, bins=numpy.arange(0., 25., 4), weights=weights, histtype='stepfilled', color='k', orientation='horizontal')
    
    plt.savefig('diapers_time.png')
    
    plt.figure()
    plt.suptitle('Number of dirty diapers per day')
    gridspec.GridSpec(3,3)

    plt.subplot2grid((3,3), (0,0), colspan=2, rowspan=3)
    date_ticks(data_diapers, label_type)
    plt.ylim([-0.5, 5.5])
    plt.plot(all_dates, n_dirty, 'ko')

    plt.subplot2grid((3,3), (0,2), colspan=1, rowspan=3)
    weights = numpy.ones_like(n_dirty)/len(n_dirty)*100.
    plt.hist(n_dirty, bins=numpy.arange(-0.5, 6.5, 1), weights=weights, histtype='stepfilled', color='k', orientation='horizontal')
    plt.ylim([-0.5, 5.5])
    plt.gca().set_yticklabels([])

    plt.savefig('diapers_number.png')


def plot_percentiles_weight_vs_length(gender, data_weight, data_length, n_months):
    
    [lengths, percentile_weight] = read_percentiles('weight-length', gender, n_months)
    percentiles = [2, 5, 10, 25, 50, 75, 90, 95, 98]
    percentile_colors = numpy.arange(0.1, 1.0, 0.1)

    plt.figure()

    for i in reversed(range(9)):
        width = 2
        if i in [1, 3, 4, 5, 7]:
            label = '{percentile}%ile'.format(percentile=percentiles[i])
            linestyle = '-'
            alpha = 1
        else:
            label = None
            linestyle = '--'
            alpha = 0.2
        plt.plot(lengths, percentile_weight[i], color=colormap(percentile_colors[i]), linestyle=linestyle, linewidth=width, label=label, alpha=alpha)

    
    plt.title('Weight vs. Length')
    plt.gca().set_xlim(left=18)
    plt.gca().set_ylim(bottom=4)
    plt.xlabel('inches')
    plt.ylabel('pounds')

    for i in range(len(data_length)-1):
        plt.plot(data_length[i][1], data_weight[i][1], 'ko')

    plt.gca().legend(loc='best')
    plt.savefig('weight-length.png')


def plot_percentiles(measurement, gender, measurement_data, n_months, official=True):
    
    plt.figure(measurement)

    if official: 
        percentile_data = read_percentiles(measurement, gender, n_months)
        percentiles = [2, 5, 10, 25, 50, 75, 90, 95, 98]
        percentile_colors = numpy.arange(0.1, 1.0, 0.1)

        for i in reversed(range(9)):
            width = 2
            if i in [1, 3, 4, 5, 7]:
                label = '{percentile}%ile'.format(percentile=percentiles[i])
                linestyle = '-'
                alpha = 1
            else:
                label = None
                linestyle = '--'
                alpha = 0.2
            plt.plot(month_dates, percentile_data[i], color=colormap(percentile_colors[i]), linestyle=linestyle, linewidth=width, label=label, alpha=alpha)

        if measurement == 'weight':
            plt.title('Weight')
            plt.gca().set_ylim(bottom=0, top=percentile_data[8][-1])
            plt.ylabel('pounds')
        elif measurement == 'length':
            plt.title('Length')
            plt.gca().set_ylim(bottom=16)
            plt.ylabel('inches')
        elif measurement == 'head':
            plt.title('Head')
            plt.gca().set_ylim(bottom=12)
            plt.ylabel('inches')
    
        date_ticks(measurement_data, label_type)
        plt.gca().legend(loc='best')

        markerstyle = 'o'
        color = 'k'
    else: 
        markerstyle = '.'
        color = 'silver'
    
    for i in range(len(measurement_data)):
        plt.plot(measurement_data[i][0], measurement_data[i][1], markerstyle, color=color)
    
    plt.savefig('{measurement}.png'.format(measurement=measurement))


def plot_proportionality(data_length, data_head, data_weight):

    plt.figure()
    plt.title('Head circumference as a fraction of length')
    date_ticks(data_head, label_type=label_type)

    plt.plot([data_head[0][0], data_head[-1][0]], [0.3, 0.3], 'k--')

    for i in range(len(data_length)-1):
        plt.plot(data_head[i][0], data_head[i][1]/data_length[i][1], 'ko')

    plt.savefig('head-length.png')

    plt.figure()
    plt.title('BMI')
    date_ticks(data_weight, label_type=label_type)

    for i in range(len(data_weight)-1):
        kg = data_weight[i][1]*0.453592
        m = data_length[i][1]*2.54/100.
        plt.plot(data_weight[i][0], kg/m**2., 'ko')

    plt.savefig('BMI.png')


def size_color_legend(size, color, label):
    
    handles = []
    for i in range(len(size)):
        handles.append(matplotlib.lines.Line2D([], [], marker='o', markersize=size[i], label=label[i], color=colormap(color[i]), linestyle='None'))
    
    return handles


if __name__ == '__main__':

    Hatch_filename = 'data_Hatch.txt'
    head_filename = 'data_head.txt'
    gender = 'girl'
    birthdate = datetime(2020, 8, 9).date()
    colormap = plt.get_cmap('viridis').reversed()  
    morning = 8
    night = 20
    label_type = 'month'

    [lines_sleeps, lines_feedings, lines_diapers, lines_weights, lines_lengths] = read_data(Hatch_filename)
    
    # parse feeding and sleep data
    data_feeding = parse_feedings(lines_feedings)
    [data_sleep, data_sleep_24] = parse_sleep(lines_sleeps)

    # set dates and ranges
    last_date = data_feeding[-1][0].date()
    n_days = (last_date - birthdate).days
    n_days_sleep = (data_sleep_24[-1][0].date() - data_sleep_24[0][0].date()).days

    all_dates = []
    for i in range(n_days+1):
        all_dates.append(birthdate + timedelta(days=i))

    data_diapers = parse_dirty_diapers(lines_diapers)

    [n_months, month_dates] = define_months()
    [data_weight, data_weight_official] = parse_weights(lines_weights, month_dates)
    data_length = parse_lengths(lines_lengths, month_dates)
    data_head = parse_head(head_filename)
    
    [bedtimes, waketimes] = plot_sleep(data_sleep, n_days_sleep)
    plot_sleep_24(data_sleep_24, bedtimes, waketimes)
    plot_feeding(data_feeding)
    plot_diapers(data_diapers)
    plot_percentiles('weight', gender, data_weight, n_months, official=False)
    plot_percentiles('weight', gender, data_weight_official, n_months)
    plot_percentiles('length', gender, data_length, n_months)
    plot_percentiles('head', gender, data_head, n_months)
    plot_percentiles_weight_vs_length(gender, data_weight_official, data_length, data_length[-2][1])
    plot_proportionality(data_length, data_head, data_weight_official)

    plt.show()
