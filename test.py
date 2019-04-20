from datetime import date
num_days = []
start_date = start_date.split()
end_date = end_date.split()
sdarr = start_date.split('-')
edarr = end_date.split('-')
f_date = date(int(sdarr[0]), int(sdarr[1]), int(sdarr[2]))
l_date = date(int(edarr[0]), int(edarr[1]), int(edarr[2]))
delta = l_date - f_date