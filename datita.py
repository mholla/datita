import string
import numpy
import math
from datetime import datetime, timedelta, date, time
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec

global birthdate, last_date, all_dates, n_days, n_days_sleep
global morning, night

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

    wdates_all = []
    weights_all = []

    for i in range(len(lines_weights)):
        data = lines_weights[i].split(',')
        wdates_all.append(parse_date(data[1]).date())
        
        weight = data[4]
        lboz = weight.split(' ')
        lb = float(lboz[0][:-2])
        if len(lboz) == 1:
            oz = 0.
        else:
            oz = float(lboz[1][:-2])
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
            data_weight[i] = [all_dates[i], data_weight[i-1][1]]

    return data_weight


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


def week_ticks(data):
    
    week_ticks = [data[0][0].date()]
    week_tick_labels = ['{month:d}/{day:d}'.format(month=week_ticks[0].month, day=week_ticks[0].day)]
    for i in range(100):
        if week_ticks[-1] + timedelta(weeks=1) < data[-1][0].date():
            new_date = week_ticks[-1] + timedelta(weeks=1)
            week_ticks.append(new_date)
            week_tick_labels.append('{month:d}/{day:d}'.format(month=new_date.month, day=new_date.day))

    plt.gca().tick_params(axis = 'x', which = 'major', labelsize = 8)
    plt.xticks(week_ticks, rotation=45)
    plt.gca().set_xticklabels(week_tick_labels)
    plt.gca().set_xlim([week_ticks[0], week_ticks[-1]])
   

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
        plt.plot(start_time, data_sleep[i][1], color=color, marker='.', markersize=8)

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
            max_sleep.append([max_sleep[-1][0] + timedelta(days=1), data_sleep[i][1]])

    plt.subplot2grid((3,3), (0,0), colspan=2, rowspan=3)
    time_ticks('x')
    plt.ylabel('hours')
    plt.yticks([0., 60., 120., 180., 240., 300., 360., 420., 480.])
    plt.gca().set_yticklabels([0, 1, 2, 3, 4, 5, 6, 7, 8])
    plt.ylim([0, 488])
    durations = []
    for i in range(len(max_sleep)):
        date = max_sleep[i][0]
        time = parse_time(max_sleep[i][0].time())
        duration = max_sleep[i][1]
        durations.append(duration)
        color = colormap(norm((date - max_sleep[0][0]).days))
        plt.plot(time, duration, 'o', color=color)


    plt.subplot2grid((3,3), (0,2), colspan=1, rowspan=3)
    plt.ylim([0, 480.])
    plt.yticks([0., 60., 120., 180., 240., 300., 360., 420., 480.])
    plt.gca().set_yticklabels([])
    weights = numpy.ones_like(durations)/len(durations)*100.
    plt.hist(durations, bins=numpy.arange(-15., 495., 30), weights=weights, histtype='stepfilled', color='k', orientation='horizontal')

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
        plt.plot(time, data_wake[i][1], 'o', color=color)
        wakes.append(data_wake[i][1])

    plt.subplot2grid((3,3), (0,2), colspan=1, rowspan=3)
    plt.ylim([0, 300.])
    plt.yticks([0., 60., 120., 180., 240., 300.])
    plt.gca().set_yticklabels([])
    weights = numpy.ones_like(wakes)/len(wakes)*100.
    plt.hist(wakes, bins=numpy.arange(-15., 315., 30), weights=weights, histtype='stepfilled', color='k', orientation='horizontal')

    plt.savefig('sleep_wake.png')


def plot_sleep_24(data_sleep_24):
    # sleep schedule with day and night distinguished
    plt.figure()
    plt.title('Sleep schedule')
    week_ticks(data_sleep_24)
    time_ticks('y')
    for i in range(len(data_sleep_24)): # 
        x = data_sleep_24[i][0].date()
        y1 = parse_time(data_sleep_24[i][0])
        y2 = parse_time(data_sleep_24[i][1])
        plt.plot([x, x], [y1, y2], 'k-', linewidth=4)
    plt.savefig('sleep_schedule.png')


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
    plt.title('Daily sleep')
    week_ticks(data_sleep_24)
    plt.ylim([0, 24.])
    plt.ylabel('hours')
    plt.plot(sleep_dates, total_sleep_24, 'k.')
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
            plt.plot(x, y, linestyle='-', color=color)

    plt.plot([0, 24], [0, 24], '--', color='white')
    plt.savefig('feeding_over_time.png')

    # feeding totals
    plt.figure()
    plt.title('Feeding totals')
    plt.ylim([0, 30.])
    plt.ylabel('ounces')
    week_ticks(data_feeding)
    plt.plot(feeding_dates, feeding_totals, 'k.')
    plt.savefig('feeding_totals.png')

    plt.figure()
    plt.title('Intake as a percentage of weight')
    week_ticks(data_feeding)
    plt.plot([feeding_dates[0], feeding_dates[-1]], [16, 16], 'k--')
    for i in range(len(feeding_dates)):
        plt.plot(feeding_dates[i], feeding_totals[i]/data_weight[i][1]/16.*100., 'k.')
    plt.savefig('feeding_percentage.png')

    # amount per feeding
    plt.figure()
    plt.title('Amount per feeding')
    week_ticks(data_feeding)
    time_ticks('y')
    
    size = [2, 4, 6, 8, 10, 12]
    color = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    label = ['1oz', '2oz', '3oz', '4oz', '5oz', '6+oz']

    for i in range(len(feeding_time)): 
        x = feeding_time[i].date()
        y = parse_time(feeding_time[i])
        z = int(math.ceil(feeding_amount[i]))
        if z > 6: 
            z = 6
        plt.plot(x, y, marker='o', markerfacecolor=colormap(color[z-1]), markeredgecolor=colormap(color[z-1]), markersize=size[z-1])
    
    handles = size_color_legend(size, color, label)
    plt.gca().legend(handles=handles, loc='lower left')    
    plt.savefig('feeding_amount_per.png')


    plt.figure()
    plt.title('Feeding amount vs. interval')
    norm = matplotlib.colors.Normalize(vmin=0, vmax=n_days)
    for i in range(1, len(feeding_time)):
        interval = (feeding_time[i] - feeding_time[i-1]).seconds/3600.
        color = colormap(norm((feeding_time[i].date() - birthdate).days))
        plt.plot(interval, feeding_amount[i], color=color, marker='.', markersize=8)
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
    plt.ylim([0, 0.3])
    week_ticks(data_feeding)
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
    week_ticks(data_diapers)
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
    week_ticks(data_diapers)
    plt.ylim([-0.5, 5.5])
    plt.plot(all_dates, n_dirty, 'ko')

    plt.subplot2grid((3,3), (0,2), colspan=1, rowspan=3)
    weights = numpy.ones_like(n_dirty)/len(n_dirty)*100.
    plt.hist(n_dirty, bins=numpy.arange(-0.5, 6.5, 1), weights=weights, histtype='stepfilled', color='k', orientation='horizontal')
    plt.ylim([-0.5, 5.5])
    plt.gca().set_yticklabels([])

    plt.savefig('diapers_number.png')


def plot_percentiles(measurement, gender, measurement_data, n_months):
    
    percentile_data = read_percentiles(measurement, gender, n_months)
    percentiles = [2, 5, 10, 25, 50, 75, 90, 95, 98]
    percentile_colors = numpy.arange(0.1, 1.0, 0.1)

    plt.figure()

    for i in reversed(range(9)):
        width = 2
        if i in [1, 3, 4, 5, 7]:
            label = percentiles[i]
            linestyle = '-'
            alpha = 1
        else:
            label = None
            linestyle = '--'
            alpha = 0.2
        plt.plot(month_dates, percentile_data[i], color=colormap(percentile_colors[i]), linestyle=linestyle, linewidth=width, label=label, alpha=alpha)

    if measurement == 'weight':
        plt.title('Weight')
        plt.plot([birthdate, last_date], [measurement_data[0][1], measurement_data[0][1] + 1.0/16.*n_days], 'k--', label='1oz per day')
        size = 1
        plt.gca().set_ylim(bottom=0)
        plt.ylabel('pounds')
    elif measurement == 'length':
        plt.title('Length')
        size = 10
        plt.gca().set_ylim(bottom=16)
        plt.ylabel('inches')
    elif measurement == 'head':
        plt.title('Head')
        size = 10
        plt.gca().set_ylim(bottom=12)
        plt.ylabel('inches')
    
    for i in range(len(measurement_data)):
        plt.plot(measurement_data[i][0], measurement_data[i][1], 'k.')

    plt.gca().legend(loc='best')
    week_ticks(measurement_data)
    plt.savefig('{measurement}.png'.format(measurement=measurement))


def size_color_legend(size, color, label):
    
    handles = []
    for i in range(len(size)):
        handles.append(matplotlib.lines.Line2D([], [], marker='o', markersize=size[i], label=label[i], markerfacecolor=colormap(color[i]), markeredgecolor=colormap(color[i]), linestyle='None'))
    
    return handles


if __name__ == '__main__':

    Hatch_filename = 'data_Hatch.txt'
    head_filename = 'data_head.txt'
    gender = 'girl'
    birthdate = datetime(2020, 8, 9).date()
    colormap = plt.get_cmap('viridis').reversed()  
    morning = 8
    night = 20

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
    data_weight = parse_weights(lines_weights, month_dates)
    data_length = parse_lengths(lines_lengths, month_dates)
    data_head = parse_head(head_filename)
    
    
    
    plot_sleep(data_sleep, n_days_sleep)
    plot_sleep_24(data_sleep_24)
    plot_feeding(data_feeding)
    plot_diapers(data_diapers)
    plot_percentiles('weight', gender, data_weight, n_months)
    plot_percentiles('length', gender, data_length, n_months)
    plot_percentiles('head', gender, data_head, n_months)

    plt.show()
