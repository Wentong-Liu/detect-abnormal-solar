def date2doy(year, month, day):
    month_leapyear = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    month_notleap = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    doy = 0

    if month == 1:
        pass
    elif year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
        for i in range(month - 1):
            doy += month_leapyear[i]
    else:
        for i in range(month - 1):
            doy += month_notleap[i]
    doy += day
    return doy


def time2sec(time):
    hour, minute, second = time.split(':')
    return (int(hour) + 11 - 24 if int(hour) + 11 > 24 else int(hour) + 11) * 3600 + int(minute) * 60 + int(second)
