import os
import threading
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

file_path = 'YOUR_PATH'
os.chdir(file_path)

# get all stock codes
# http://www.sse.com.cn/assortment/stock/list/share/
# http://www.szse.cn/market/product/stock/list/index.html
sz_code = pd.read_excel("SZA.xlsx", usecols=['公司代码', 'A股上市日期'])
sz_code['flag'] = '1'
sh_code = pd.read_csv("SHA.csv", names=['上市日期']).reset_index().iloc[:, :2]
sh_code.columns = ['公司代码', 'A股上市日期']
sh_code['flag'] = '0'
code_all = pd.concat((sz_code, sh_code)).values


def get_data(code_list, k):
    for i, (code, start_date, flag) in enumerate(code_list):
        code = str(code).zfill(6)
        start_date = str(start_date).strip().replace('-', '')
        url = 'ADDRESS_MASKED?code=%s&start=%s&end=20201231&fields=TCLOSE;HIGH;LOW;TOPEN;LCLOSE;CHG;PCHG;TURNOVER;VOTURNOVER;VATURNOVER;TCAP;MCAP' % (
            flag + code, start_date)
        filename = '%s.csv' % code
        urlretrieve(url, filename)
        if k == 0 & i % 10 == 0:
            print(i + 1, '/', len(code_list))


def parallel_run(func, file_list, threads):
    steps = int(len(file_list) / (threads)) + 1
    num = int(len(file_list) / steps) + 1
    for k in range(num):
        name = 't' + str(k)
        locals()['t' + str(k)] = threading.Thread(
            target=func,
            args=(file_list[steps * k:min(steps * (k + 1), len(file_list))],
                  k))
    for k in range(num):
        locals()['t' + str(k)].start()
    for k in range(num):
        locals()['t' + str(k)].join()
    print('all done!')


parallel_run(get_data, code_all, 50)  # download data

# merging
files = os.listdir()
files = [x for x in files if 'S' not in x]  # skip SHA.csv and SZA.csv
res = []
failed = []
for i, file in enumerate(files):
    try:
        with open(file, encoding='GB2312') as f:
            data = f.read()
            data = data.split('\n')[1:-1]
            if len(data) > 250:
                data = data[:-125]  # skip the first 6 months
                data = list(map(lambda x: np.array(x.split(',')), data))
                data = list(
                    filter(lambda x: int(x[0].split('-')[0]) > 2016, data))
                data = list(
                    map(lambda x: x[[0, 1, 3, 9, 11, 13]], data)
                )  # keep date, code, close price, daily return, volume and total cap
                res.extend(data)
    except:
        failed.append(file)
    if i % 50 == 0: print(i)
print('All done, failed files:\n', failed)

columns = [
    'date', 'code', 'close_price', 'daily_return', 'trade_volume',
    'liquid_market_value'
]
res = pd.DataFrame(res, columns=columns)
res.to_csv('merged_data_all.csv', index=False)

# generate x and y
df1 = pd.read_csv('merged_data_all.csv')

df1['code'] = df1.code.map(lambda x: x.replace("'", "")).astype(str)
df1['date'] = pd.to_datetime(df1['date'])
df1 = df1[df1.daily_return != 'None']
df1['daily_return'] = df1['daily_return'].astype(float)
df1.sort_values(['code', 'date'], inplace=True)


def single_stock_sample_generator(df):
    """
    for a 33-days-data-block
    we first check if there is any abnormal things exsits
    then we generate x and y for this block
    """
    liquid_check = df.trade_volume.min(
    ) == 0  # skip stocks that are not trading during the selected days
    market_cap_check = df.liquid_market_value.min(
    ) < 9.75 * 10**8  # skip stocks whose liquid_market_value is too small, those stocks have very high volatility and will influence data distribution
    price_check = df.close_price.min(
    ) < 10  # skip stocks whose price is too low, since the daily return is our target and the minimum price change is 0.01, stock's return rate will be influenced a lot by a 0.01 change for low price stock
    if liquid_check or market_cap_check or price_check:
        return np.array((None, None))

    x_temp = df.iloc[:30]['daily_return'].values
    yesterday_price = df.iloc[29]['close_price'].values
    target_price = df.iloc[-1]['close_price'].values
    y_temp = (target_price - yesterday_price) / yesterday_price * 10
    return np.array((x_temp, y_temp))


def all_stock_generator(df):
    """
    for a single stock
    we seperate every 33 days data (30 history data + 3 forcast data)
    into a new dataframe
    """
    df['group_no'] = list(range(len(df)))
    df['group_no'] = df['group_no'] // 33
    df = df[df.group_no < max(df.group_no)]
    res = df.groupby('group_no').apply(single_stock_sample_generator).values
    return np.array((res))


import time

tik = time.time()
res = df1.groupby('code').apply(
    all_stock_generator
)  # apply it to all stocks, the result is a list of 3600 stocks' list which contains the x and y of each stock
print(time.time() - tik)
del df1

from itertools import chain

res = list(
    chain.from_iterable(res))  # transform a lists of list to just a list
res = list(filter(lambda x: np.any(x[0] != None), res))

np.save('all_x_y', res)
